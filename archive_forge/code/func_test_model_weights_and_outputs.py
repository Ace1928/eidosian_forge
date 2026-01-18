import os
import argparse
import sys
import logging
import mxnet as mx
from convert_caffe_modelzoo import convert_caffe_model, get_model_meta_info, download_caffe_model
from compare_layers import convert_and_compare_caffe_to_mxnet
from test_score import download_data  # pylint: disable=wrong-import-position
from score import score # pylint: disable=wrong-import-position
def test_model_weights_and_outputs(model_name, image_url, gpu):
    """
    Run the layer comparison on one of the known caffe models.
    :param model_name: available models are listed in convert_caffe_modelzoo.py
    :param image_url: image file or url to run inference on
    :param gpu: gpu to use, -1 for cpu
    """
    logging.info('test weights and outputs of model: %s', model_name)
    meta_info = get_model_meta_info(model_name)
    prototxt, caffemodel, mean = download_caffe_model(model_name, meta_info, dst_dir='./model')
    convert_and_compare_caffe_to_mxnet(image_url, gpu, prototxt, caffemodel, mean, mean_diff_allowed=0.001, max_diff_allowed=0.1)