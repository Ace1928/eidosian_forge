import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_equal')
def convert_broadcast_equal(node, **kwargs):
    """Map MXNet's broadcast_equal operator attributes to onnx's Equal operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    nodes = [make_node('Equal', input_nodes, [name + '_equal']), make_node('Cast', [name + '_equal'], [name], name=name, to=int(dtype_t))]
    return nodes