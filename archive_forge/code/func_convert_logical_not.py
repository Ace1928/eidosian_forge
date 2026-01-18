import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('logical_not')
def convert_logical_not(node, **kwargs):
    """Map MXNet's logical not operator attributes to onnx's Not operator
    and return the created node.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, _ = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    nodes = [make_node('Cast', [input_nodes[0]], [name + '_cast'], to=int(TensorProto.BOOL)), make_node('Not', [name + '_cast'], [name + '_not']), make_node('Cast', [name + '_not'], [name], name=name, to=int(dtype_t))]
    return nodes