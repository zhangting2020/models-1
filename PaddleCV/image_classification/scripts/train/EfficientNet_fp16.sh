#!/bin/bash -ex
export GLOG_v=1
#export GLOG_vmodule=operator=3
export CUDA_VISIBLE_DEVICES=4
export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_exhaustive_search=0
export FLAGS_cudnn_batchnorm_spatial_persistent=1

DATA_DIR="/data/ILSVRC2012/"

DATA_FORMAT="NHWC"
USE_FP16=true #whether to use float16
USE_DALI=true
USE_ADDTO=true

if ${USE_ADDTO} ;then
    export FLAGS_max_inplace_grad_add=8
fi

if ${USE_DALI}; then
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
fi

python -u train.py \
       --model=EfficientNet \
       --data_dir=${DATA_DIR} \
       --batch_size=128 \
       --image_shape 4 224 224 \
       --test_batch_size=128 \
       --resize_short_size=256 \
       --model_save_dir=output/ \
       --lr_strategy=exponential_decay_warmup \
       --lr=0.032 \
       --num_epochs=360 \
       --l2_decay=1e-5 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1 \
       --use_ema=True \
       --ema_decay=0.9999 \
       --drop_connect_rate=0.1 \
       --padding_type="SAME" \
       --interpolation=2 \
       --use_fp16=${USE_FP16} \
       --scale_loss=128.0 \
       --use_dynamic_loss_scaling=true \
       --data_format=${DATA_FORMAT} \
       --fuse_bn_act_ops=true \
       --fuse_bn_add_act_ops=true \
       --fuse_elewise_add_act_ops=true \
       --enable_addto=${USE_ADDTO} \
       --use_dali=${USE_DALI} 
