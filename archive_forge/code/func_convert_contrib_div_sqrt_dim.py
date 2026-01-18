import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_contrib_div_sqrt_dim')
def convert_contrib_div_sqrt_dim(node, **kwargs):
    """Map MXNet's _contrib_div_sqrt_dim operator
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([1], name + '_1_f', kwargs['initializer'], dtype=dtype)
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Shape', [name + '_shape'], [name + '_dim']), make_node('Sub', [name + '_dim', name + '_1'], [name + '_dim_m1']), make_node('Slice', [name + '_shape', name + '_dim_m1', name + '_dim', name + '_0'], [name + '_c_']), make_node('Cast', [name + '_c_'], [name + '_c'], to=dtype_t), make_node('Sqrt', [name + '_c'], [name + '_c_sqrt']), make_node('Div', [name + '_1_f', name + '_c_sqrt'], [name + '_1_over_c_sqrt']), make_node('Mul', [input_nodes[0], name + '_1_over_c_sqrt'], [name])]
    return nodes