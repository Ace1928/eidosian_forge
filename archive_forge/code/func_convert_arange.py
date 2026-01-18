import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_arange')
def convert_arange(node, **kwargs):
    """Map MXNet's arange operator attributes to onnx's Range operator.
    """
    from onnx.helper import make_node
    name, _, attrs = get_inputs(node, kwargs)
    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')
    start = attrs.get('start', 0.0)
    stop = attrs.get('stop')
    step = attrs.get('step', 1.0)
    dtype = attrs.get('dtype', 'float32')
    repeat = int(attrs.get('repeat', 1))
    if stop == 'None':
        stop = start
        start = 0
    if repeat != 1:
        raise NotImplementedError('arange operator with repeat != 1 not yet implemented.')
    create_const_scalar_node(name + '_start', np.dtype(dtype).type(start), kwargs)
    create_const_scalar_node(name + '_stop', np.dtype(dtype).type(stop), kwargs)
    create_const_scalar_node(name + '_step', np.dtype(dtype).type(step), kwargs)
    nodes = [make_node('Range', [name + '_start', name + '_stop', name + '_step'], [name], name=name)]
    return (nodes, (dtype,))