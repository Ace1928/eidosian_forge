import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_mod')
def convert_broadcast_mod(node, **kwargs):
    """Map MXNet's broadcast_mod operator
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)
    nodes = [make_node('Sub', [input_nodes[1], input_nodes[1]], [name + '_zero']), make_node('Mod', [input_nodes[0], input_nodes[1]], [name + '_mod'], fmod=1), make_node('Less', [input_nodes[0], name + '_zero'], [name + '_mask_0']), make_node('Less', [input_nodes[1], name + '_zero'], [name + '_mask_1']), make_node('Equal', [name + '_mod', name + '_zero'], [name + '_mask_2_']), make_node('Not', [name + '_mask_2_'], [name + '_mask_2']), make_node('Xor', [name + '_mask_0', name + '_mask_1'], [name + '_mask_']), make_node('And', [name + '_mask_', name + '_mask_2'], [name + '_mask']), make_node('Where', [name + '_mask', input_nodes[1], name + '_zero'], [name + '_adjustment']), make_node('Add', [name + '_mod', name + '_adjustment'], [name + '_adjusted']), make_node('Equal', [input_nodes[1], name + '_zero'], [name + '_mask_div_0']), make_node('Where', [name + '_mask_div_0', name + '_zero', name + '_adjusted'], [name], name=name)]
    return nodes