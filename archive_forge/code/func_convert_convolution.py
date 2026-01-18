import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Convolution')
def convert_convolution(node, **kwargs):
    """Map MXNet's convolution operator attributes to onnx's Conv operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    kernel = convert_string_to_list(attrs.get('kernel', '()'))
    stride = convert_string_to_list(attrs.get('stride', '()'))
    dilate = convert_string_to_list(attrs.get('dilate', '()'))
    pad = convert_string_to_list(attrs.get('pad', '()'))
    num_group = int(attrs.get('num_group', 1))
    no_bias = attrs.get('no_bias', 'False')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NCDHW']:
        raise NotImplementedError("Convolution currently does not support layout not in ['NCHW', 'NCDHW']")
    if no_bias == 'True':
        assert len(input_nodes) == 2, 'Convolution takes 2 input if no_bias==True'
    else:
        assert len(input_nodes) == 3, 'Convolution takes 3 input if no_bias==False'
    kwargs_ = {}
    if kernel:
        kwargs_['kernel_shape'] = tuple(kernel)
    if pad:
        kwargs_['pads'] = tuple(pad) + tuple(pad)
    if stride:
        kwargs_['strides'] = stride
    if dilate:
        kwargs_['dilations'] = dilate
    nodes = [make_node('Conv', input_nodes, [name], group=num_group, **kwargs_)]
    return nodes