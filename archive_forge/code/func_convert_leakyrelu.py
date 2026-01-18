import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('LeakyReLU')
def convert_leakyrelu(node, **kwargs):
    """Map MXNet's LeakyReLU operator attributes to onnx's Elu/LeakyRelu/PRelu operators
    based on the input node's attributes and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    act_type = attrs.get('act_type', 'leaky')
    alpha = float(attrs.get('slope', 0.25))
    act_name = {'elu': 'Elu', 'leaky': 'LeakyRelu', 'prelu': 'PRelu', 'selu': 'Selu'}
    if act_type in ('prelu', 'selu'):
        node = onnx.helper.make_node(act_name[act_type], inputs=input_nodes, outputs=[name], name=name)
    elif act_type in ('gelu',):
        sqrt2 = np.float32(1.4142135623730951)
        create_const_scalar_node(name + '_sqrt2', sqrt2, kwargs)
        create_const_scalar_node(name + '_one', np.float32(1.0), kwargs)
        create_const_scalar_node(name + '_half', np.float32(0.5), kwargs)
        nodes = [make_node('Div', [input_nodes[0], name + '_sqrt2'], [name + '_div0_out']), make_node('Erf', [name + '_div0_out'], [name + '_erf0_out']), make_node('Add', [name + '_erf0_out', name + '_one'], [name + '_add0_out']), make_node('Mul', [input_nodes[0], name + '_add0_out'], [name + '_mul0_out']), make_node('Mul', [name + '_mul0_out', name + '_half'], [name], name=name)]
        return nodes
    else:
        node = onnx.helper.make_node(act_name[act_type], inputs=input_nodes, outputs=[name], name=name, alpha=alpha)
    return [node]