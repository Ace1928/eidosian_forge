import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_random_normal')
def convert_random_normal(node, **kwargs):
    """Map MXNet's random_normal operator attributes to onnx's RandomNormal
    operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    mean = float(attrs.get('loc', 0))
    scale = float(attrs.get('scale', 1.0))
    shape = convert_string_to_list(attrs.get('shape', '[]'))
    dtype = np.dtype(attrs.get('dtype', 'float32'))
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    node = onnx.helper.make_node('RandomNormal', input_nodes, [name], mean=mean, scale=scale, dtype=dtype_t, shape=shape, name=name)
    return ([node], (dtype,))