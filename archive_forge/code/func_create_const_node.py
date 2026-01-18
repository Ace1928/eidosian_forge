import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
def create_const_node(input_name, value, kwargs):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    from onnx.helper import make_tensor
    initializer = kwargs['initializer']
    dtype = value.dtype
    if dtype == 'float16':
        value = np.float16(value).view(np.uint16)
    input_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    input_shape = value.shape
    tensor_node = make_tensor(input_name, input_type, input_shape, value)
    initializer.append(tensor_node)