import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('dot')
def convert_dot(node, **kwargs):
    """Map MXNet's dot operator attributes to onnx's
    MatMul and Transpose operators based on the values set for
    transpose_a, transpose_b attributes."""
    logging.warning('Converting dot operator... Please note that due to ONNX limitation, the behavior for when inputs > 2-D is different from that of MXNet dot.')
    name, inputs, attrs = get_inputs(node, kwargs)
    trans_a = get_boolean_attribute_value(attrs, 'transpose_a')
    trans_b = get_boolean_attribute_value(attrs, 'transpose_b')
    nodes = []
    input_nodes = []
    if trans_a:
        nodes.append(create_helper_trans_node(name + '_a', inputs[0]))
        input_nodes.append(name + '_a')
    else:
        input_nodes.append(inputs[0])
    if trans_b:
        nodes.append(create_helper_trans_node(name + '_b', inputs[1]))
        input_nodes.append(name + '_b')
    else:
        input_nodes.append(inputs[1])
    nodes.append(onnx.helper.make_node('MatMul', input_nodes, [name], name=name))
    return nodes