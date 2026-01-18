import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Cast')
def convert_cast(node, **kwargs):
    """Map MXNet's Cast operator attributes to onnx's Cast operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dtype = np.dtype(attrs.get('dtype'))
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    nodes = [onnx.helper.make_node('Cast', input_nodes, [name], to=dtype_t, name=name)]
    return (nodes, (dtype,))