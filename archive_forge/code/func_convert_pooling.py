import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Pooling')
def convert_pooling(node, **kwargs):
    """Map MXNet's Pooling operator attributes to onnx's
    MaxPool/AveragePool/GlobalMaxPool/GlobalAveragePool operators
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    kernel = convert_string_to_list(attrs.get('kernel', '()'))
    pool_type = attrs.get('pool_type', 'max')
    global_pool = attrs.get('global_pool', 'False')
    _ = attrs.get('cudnn_off', 'False')
    pooling_convention = attrs.get('pooling_convention', 'valid')
    stride = convert_string_to_list(attrs.get('stride', '()'))
    pad = convert_string_to_list(attrs.get('pad', '()'))
    p_value = int(attrs.get('p_value', '0'))
    count_include_pad = attrs.get('count_include_pad', 'True')
    layout = attrs.get('layout', 'NCHW')
    if pooling_convention == 'same':
        raise NotImplementedError("Pooling currently does not support pooling_convention=='same'")
    if pool_type == 'sum':
        raise NotImplementedError("Pooling currently does not support pool_type=='sum'")
    if pool_type == 'lp' and global_pool == 'False' and (pooling_convention != 'valid'):
        raise NotImplementedError("Pooling currently does not support pooling_convention!='valid' when pool_type=='lp' and global_pool==False")
    if layout not in ['NCHW', 'NCDHW']:
        raise NotImplementedError("Pooling currently does not support layout not in ['NCHW', 'NCDHW']")
    kwargs_ = {}
    if kernel:
        kwargs_['kernel_shape'] = tuple(kernel)
    if pad:
        kwargs_['pads'] = tuple(pad) + tuple(pad)
    if stride:
        kwargs_['strides'] = stride
    ceil_mode = 1 if pooling_convention == 'full' else 0
    count_include_pad = 1 if count_include_pad == 'True' else 0
    nodes = []
    if pool_type == 'avg' and global_pool == 'False':
        nodes += [make_node('AveragePool', [input_nodes[0]], [name], ceil_mode=ceil_mode, count_include_pad=count_include_pad, **kwargs_)]
    elif pool_type == 'max' and global_pool == 'False':
        nodes += [make_node('MaxPool', [input_nodes[0]], [name], ceil_mode=ceil_mode, **kwargs_)]
    elif pool_type == 'lp' and global_pool == 'False':
        nodes += [make_node('LpPool', [input_nodes[0]], [name], p=p_value, **kwargs_)]
    elif pool_type == 'avg' and global_pool == 'True':
        nodes += [make_node('GlobalAveragePool', [input_nodes[0]], [name])]
    elif pool_type == 'max' and global_pool == 'True':
        nodes += [make_node('GlobalMaxPool', [input_nodes[0]], [name])]
    elif pool_type == 'lp' and global_pool == 'True':
        nodes += [make_node('GlobalLpPool', [input_nodes[0]], [name], p=p_value)]
    else:
        raise NotImplementedError('Unknown parameter values in Pooling')
    return nodes