import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('zeros_like')
def convert_zeros_like(node, **kwargs):
    """Map MXNet's zeros_like operator attributes to onnx's ConstantOfShape operator.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, _ = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = np.dtype(input_dtypes[0])
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    tensor_value = make_tensor(name + '_zero', dtype_t, [1], [0])
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('ConstantOfShape', [name + '_shape'], [name], name=name, value=tensor_value)]
    return nodes