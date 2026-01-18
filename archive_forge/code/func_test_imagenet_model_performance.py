import os
import argparse
import sys
import logging
import mxnet as mx
from convert_caffe_modelzoo import convert_caffe_model, get_model_meta_info, download_caffe_model
from compare_layers import convert_and_compare_caffe_to_mxnet
from test_score import download_data  # pylint: disable=wrong-import-position
from score import score # pylint: disable=wrong-import-position
def test_imagenet_model_performance(model_name, val_data, gpus, batch_size):
    """test model performance on imagenet """
    logging.info('test performance of model: %s', model_name)
    meta_info = get_model_meta_info(model_name)
    [model_name, mean] = convert_caffe_model(model_name, meta_info)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)
    acc = [mx.metric.create('acc'), mx.metric.create('top_k_accuracy', top_k=5)]
    if isinstance(mean, str):
        mean_args = {'mean_img': mean}
    else:
        mean_args = {'rgb_mean': ','.join([str(i) for i in mean])}
    print(val_data)
    gpus_string = '' if gpus[0] == -1 else ','.join([str(i) for i in gpus])
    speed, = score(model=(sym, arg_params, aux_params), data_val=val_data, label_name='prob_label', metrics=acc, gpus=gpus_string, batch_size=batch_size, max_num_examples=500, **mean_args)
    logging.info('speed : %f image/sec', speed)
    for a in acc:
        logging.info(a.get())
    max_performance_diff_allowed = 0.03
    assert acc[0].get()[1] > meta_info['top-1-acc'] - max_performance_diff_allowed
    assert acc[1].get()[1] > meta_info['top-5-acc'] - max_performance_diff_allowed