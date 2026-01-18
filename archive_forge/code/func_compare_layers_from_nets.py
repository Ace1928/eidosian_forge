import argparse
import logging
import os
import warnings
import numpy as np
import cv2
import mxnet as mx
def compare_layers_from_nets(caffe_net, arg_params, aux_params, exe, layer_name_to_record, top_to_layers, mean_diff_allowed, max_diff_allowed):
    """
    Compare layer by layer of a caffe network with mxnet network
    :param caffe_net: loaded caffe network
    :param arg_params: arguments
    :param aux_params: auxiliary parameters
    :param exe: mxnet model
    :param layer_name_to_record: map between caffe layer and information record
    :param top_to_layers: map between caffe blob name to layers which outputs it (including inplace)
    :param mean_diff_allowed: mean difference allowed between caffe blob and mxnet blob
    :param max_diff_allowed: max difference allowed between caffe blob and mxnet blob
    """
    import re
    log_format = '  {0:<40}  {1:<40}  {2:<8}  {3:>10}  {4:>10}  {5:<1}'
    compare_layers_from_nets.is_first_convolution = True

    def _compare_blob(caf_blob, mx_blob, caf_name, mx_name, blob_type, note):
        diff = np.abs(mx_blob - caf_blob)
        diff_mean = diff.mean()
        diff_max = diff.max()
        logging.info(log_format.format(caf_name, mx_name, blob_type, '%4.5f' % diff_mean, '%4.5f' % diff_max, note))
        assert diff_mean < mean_diff_allowed
        assert diff_max < max_diff_allowed

    def _process_layer_parameters(layer):
        logging.debug('processing layer %s of type %s', layer.name, layer.type)
        normalized_layer_name = re.sub('[-/]', '_', layer.name)
        if layer.name in caffe_net.params and layer.type in ['Convolution', 'InnerProduct', 'Deconvolution']:
            has_bias = len(caffe_net.params[layer.name]) > 1
            mx_name_weight = '{}_weight'.format(normalized_layer_name)
            mx_beta = arg_params[mx_name_weight].asnumpy()
            if layer.type == 'Convolution' and compare_layers_from_nets.is_first_convolution:
                compare_layers_from_nets.is_first_convolution = False
                if mx_beta.shape[1] == 3 or mx_beta.shape[1] == 4:
                    mx_beta[:, [0, 2], :, :] = mx_beta[:, [2, 0], :, :]
            caf_beta = caffe_net.params[layer.name][0].data
            _compare_blob(caf_beta, mx_beta, layer.name, mx_name_weight, 'weight', '')
            if has_bias:
                mx_name_bias = '{}_bias'.format(normalized_layer_name)
                mx_gamma = arg_params[mx_name_bias].asnumpy()
                caf_gamma = caffe_net.params[layer.name][1].data
                _compare_blob(caf_gamma, mx_gamma, layer.name, mx_name_bias, 'bias', '')
        elif layer.name in caffe_net.params and layer.type == 'Scale':
            if 'scale' in normalized_layer_name:
                bn_name = normalized_layer_name.replace('scale', 'bn')
            elif 'sc' in normalized_layer_name:
                bn_name = normalized_layer_name.replace('sc', 'bn')
            else:
                assert False, 'Unknown name convention for bn/scale'
            beta_name = '{}_beta'.format(bn_name)
            gamma_name = '{}_gamma'.format(bn_name)
            mx_beta = arg_params[beta_name].asnumpy()
            caf_beta = caffe_net.params[layer.name][1].data
            _compare_blob(caf_beta, mx_beta, layer.name, beta_name, 'mov_mean', '')
            mx_gamma = arg_params[gamma_name].asnumpy()
            caf_gamma = caffe_net.params[layer.name][0].data
            _compare_blob(caf_gamma, mx_gamma, layer.name, gamma_name, 'mov_var', '')
        elif layer.name in caffe_net.params and layer.type == 'BatchNorm':
            mean_name = '{}_moving_mean'.format(normalized_layer_name)
            var_name = '{}_moving_var'.format(normalized_layer_name)
            caf_rescale_factor = caffe_net.params[layer.name][2].data
            mx_mean = aux_params[mean_name].asnumpy()
            caf_mean = caffe_net.params[layer.name][0].data / caf_rescale_factor
            _compare_blob(caf_mean, mx_mean, layer.name, mean_name, 'mean', '')
            mx_var = aux_params[var_name].asnumpy()
            caf_var = caffe_net.params[layer.name][1].data / caf_rescale_factor
            _compare_blob(caf_var, mx_var, layer.name, var_name, 'var', 'expect 1e-04 change due to cudnn eps')
        elif layer.type in ['Input', 'Pooling', 'ReLU', 'Eltwise', 'Softmax', 'LRN', 'Concat', 'Dropout', 'Crop']:
            pass
        else:
            warnings.warn('No handling for layer %s of type %s, should we ignore it?', layer.name, layer.type)

    def _process_layer_output(caffe_blob_name):
        logging.debug('processing blob %s', caffe_blob_name)
        if caffe_blob_name not in top_to_layers:
            return
        caf_blob = caffe_net.blobs[caffe_blob_name].data
        if caffe_blob_name == 'data':
            if caf_blob.shape[1] == 3 or caf_blob.shape[1] == 4:
                caf_blob[:, [0, 2], :, :] = caf_blob[:, [2, 0], :, :]
            mx_name = 'data'
        else:
            last_layer_name = top_to_layers[caffe_blob_name][-1]
            normalized_last_layer_name = re.sub('[-/]', '_', last_layer_name)
            mx_name = '{}_output'.format(normalized_last_layer_name)
            if 'scale' in mx_name:
                mx_name = mx_name.replace('scale', 'bn')
            elif 'sc' in mx_name:
                mx_name = mx_name.replace('sc', 'bn')
        if mx_name not in exe.output_dict:
            logging.error('mxnet blob %s is missing, time to extend the compare tool..', mx_name)
            return
        mx_blob = exe.output_dict[mx_name].asnumpy()
        _compare_blob(caf_blob, mx_blob, caffe_blob_name, mx_name, 'output', '')
        return
    logging.info('\n***** Network Parameters '.ljust(140, '*'))
    logging.info(log_format.format('CAFFE', 'MXNET', 'Type', 'Mean(diff)', 'Max(diff)', 'Note'))
    first_layer_name = layer_name_to_record.keys()[0]
    _bfs(layer_name_to_record[first_layer_name], _process_layer_parameters)
    logging.info('\n***** Network Outputs '.ljust(140, '*'))
    logging.info(log_format.format('CAFFE', 'MXNET', 'Type', 'Mean(diff)', 'Max(diff)', 'Note'))
    for caffe_blob_name in caffe_net.blobs.keys():
        _process_layer_output(caffe_blob_name)