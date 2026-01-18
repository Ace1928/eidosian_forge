import argparse
import logging
import os
import warnings
import numpy as np
import cv2
import mxnet as mx
def convert_and_compare_caffe_to_mxnet(image_url, gpu, caffe_prototxt_path, caffe_model_path, caffe_mean, mean_diff_allowed, max_diff_allowed):
    """
    Run the layer comparison on a caffe model, given its prototxt, weights and mean.
    The comparison is done by inferring on a given image using both caffe and mxnet model
    :param image_url: image file or url to run inference on
    :param gpu: gpu to use, -1 for cpu
    :param caffe_prototxt_path: path to caffe prototxt
    :param caffe_model_path: path to caffe weights
    :param caffe_mean: path to caffe mean file
    """
    import caffe
    from caffe_proto_utils import read_network_dag, process_network_proto, read_caffe_mean
    from convert_model import convert_model
    if isinstance(caffe_mean, str):
        caffe_mean = read_caffe_mean(caffe_mean)
    elif caffe_mean is None:
        pass
    elif len(caffe_mean) == 3:
        caffe_mean = caffe_mean[::-1]
    caffe_root = os.path.dirname(os.path.dirname(caffe.__path__[0]))
    caffe_prototxt_path = process_network_proto(caffe_root, caffe_prototxt_path)
    _, layer_name_to_record, top_to_layers = read_network_dag(caffe_prototxt_path)
    caffe.set_mode_cpu()
    caffe_net = caffe.Net(caffe_prototxt_path, caffe_model_path, caffe.TEST)
    image_dims = tuple(caffe_net.blobs['data'].shape)[2:4]
    logging.info('getting image %s', image_url)
    img_rgb = read_image(image_url, image_dims, caffe_mean)
    img_bgr = img_rgb[:, ::-1, :, :]
    caffe_net.blobs['data'].reshape(*img_bgr.shape)
    caffe_net.blobs['data'].data[...] = img_bgr
    _ = caffe_net.forward()
    sym, arg_params, aux_params, _ = convert_model(caffe_prototxt_path, caffe_model_path)
    sym = sym.get_internals()
    if gpu < 0:
        ctx = mx.cpu(0)
    else:
        ctx = mx.gpu(gpu)
    arg_params, aux_params = _ch_dev(arg_params, aux_params, ctx)
    arg_params['data'] = mx.nd.array(img_rgb, ctx)
    arg_params['prob_label'] = mx.nd.empty((1,), ctx)
    exe = sym.bind(ctx, arg_params, args_grad=None, grad_req='null', aux_states=aux_params)
    exe.forward(is_train=False)
    compare_layers_from_nets(caffe_net, arg_params, aux_params, exe, layer_name_to_record, top_to_layers, mean_diff_allowed, max_diff_allowed)