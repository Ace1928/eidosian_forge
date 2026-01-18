import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Deconvolution')
def convert_deconvolution(node, **kwargs):
    """Map MXNet's deconvolution operator attributes to onnx's ConvTranspose operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    kernel_shape = convert_string_to_list(attrs.get('kernel', '()'))
    strides = convert_string_to_list(attrs.get('stride', '()'))
    pads = convert_string_to_list(attrs.get('pad', '()'))
    group = int(attrs.get('num_group', 1))
    dilations = convert_string_to_list(attrs.get('dilate', '()'))
    output_padding = convert_string_to_list(attrs.get('adj', '()'))
    layout = attrs.get('layout', 'NCHW')
    target_shape = attrs.get('target_shape', '')
    no_bias = attrs.get('no_bias', 'False')
    pads = pads + pads
    if target_shape != '':
        raise NotImplementedError('Deconvolution currently does not support target_shape')
    if layout not in ['NCHW', 'NCDHW', 'NCW']:
        raise NotImplementedError("Deconvolution currently does not support layout not in ['NCHW', 'NCDHW', 'NCW']")
    if no_bias == 'True':
        assert len(input_nodes) == 2, 'Deconvolution takes 2 input if no_bias==True'
    else:
        assert len(input_nodes) == 3, 'Deconvolution takes 3 input if no_bias==False'
    kwargs_ = {}
    if kernel_shape:
        kwargs_['kernel_shape'] = kernel_shape
    if pads:
        kwargs_['pads'] = pads
    if strides:
        kwargs_['strides'] = strides
    if dilations:
        kwargs_['dilations'] = dilations
    if output_padding:
        kwargs_['output_padding'] = output_padding
    deconv_node = onnx.helper.make_node('ConvTranspose', inputs=input_nodes, outputs=[name], group=group, **kwargs_)
    return [deconv_node]