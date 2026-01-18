import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_random_uniform')
def convert_random_uniform(node, **kwargs):
    """Map MXNet's random_uniform operator attributes to onnx's RandomUniform
    operator and return the created node.
    """
    name, _, attrs = get_inputs(node, kwargs)
    low = float(attrs.get('low', 0))
    high = float(attrs.get('high', 1.0))
    shape = convert_string_to_list(attrs.get('shape', '[]'))
    dtype = np.dtype(attrs.get('dtype', 'float32'))
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    node = onnx.helper.make_node('RandomUniform', [], [name], low=low, high=high, dtype=dtype_t, shape=shape, name=name)
    return ([node], (dtype,))