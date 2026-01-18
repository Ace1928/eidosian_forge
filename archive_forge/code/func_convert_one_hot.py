import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('one_hot')
def convert_one_hot(node, **kwargs):
    """Map MXNet's one_hot operator attributes to onnx's OneHot operator
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    depth = int(attrs.get('depth'))
    on_value = float(attrs.get('on_value', 1.0))
    off_value = float(attrs.get('off_value', 0.0))
    dtype = attrs.get('dtype', 'float32')
    create_tensor([off_value, on_value], name + '_values', kwargs['initializer'], dtype=np.dtype(dtype))
    create_tensor([depth], name + '_depth', kwargs['initializer'])
    nodes = [make_node('Cast', [input_nodes[0]], [name + '_cast'], to=int(TensorProto.INT64)), make_node('OneHot', [name + '_cast', name + '_depth', name + '_values'], [name], name=name)]
    return nodes