import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_contrib_interleaved_matmul_selfatt_qk')
def convert_matmul_selfatt_qk(node, **kwargs):
    """Map MXNet's _contrib_interleaved_matmul_selfatt_qk operator
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    heads = int(attrs.get('heads'))
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([1], name + '_1_f', kwargs['initializer'], dtype='float32')
    create_tensor([2], name + '_2', kwargs['initializer'])
    create_tensor([3], name + '_3', kwargs['initializer'])
    create_tensor([heads], name + '_c', kwargs['initializer'])
    create_tensor([3], name + '_d', kwargs['initializer'])
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_data_shape']), make_node('Slice', [name + '_data_shape', name + '_0', name + '_1'], [name + '_a']), make_node('Slice', [name + '_data_shape', name + '_1', name + '_2'], [name + '_b']), make_node('Slice', [name + '_data_shape', name + '_2', name + '_3'], [name + '_cde']), make_node('Div', [name + '_cde', name + '_c'], [name + '_de']), make_node('Div', [name + '_de', name + '_d'], [name + '_e']), make_node('Cast', [name + '_e'], [name + '_e_f'], to=int(TensorProto.FLOAT)), make_node('Sqrt', [name + '_e_f'], [name + '_sqrt_e']), make_node('Div', [name + '_1_f', name + '_sqrt_e'], [name + '_1_over_sqrt_e']), make_node('Mul', [name + '_b', name + '_c'], [name + '_bc']), make_node('Concat', [name + '_a', name + '_b', name + '_c', name + '_d', name + '_e'], [name + '_shape0'], axis=0), make_node('Concat', [name + '_0', name + '_0', name + '_0', name + '_0', name + '_0'], [name + '_slice_start0'], axis=0), make_node('Concat', [name + '_a', name + '_b', name + '_c', name + '_1', name + '_e'], [name + '_slice_end0'], axis=0), make_node('Concat', [name + '_a', name + '_b', name + '_c', name + '_e'], [name + '_shape1'], axis=0), make_node('Concat', [name + '_bc', name + '_a', name + '_e'], [name + '_shape2'], axis=0), make_node('Concat', [name + '_0', name + '_0', name + '_0', name + '_1', name + '_0'], [name + '_slice_start1'], axis=0), make_node('Concat', [name + '_a', name + '_b', name + '_c', name + '_2', name + '_e'], [name + '_slice_end1'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape0'], [name + '_reshape0_out']), make_node('Slice', [name + '_reshape0_out', name + '_slice_start0', name + '_slice_end0'], [name + '_slice0_out']), make_node('Reshape', [name + '_slice0_out', name + '_shape1'], [name + '_reshape1_out']), make_node('Transpose', [name + '_reshape1_out'], [name + '_transpose0_out'], perm=(1, 2, 0, 3)), make_node('Reshape', [name + '_transpose0_out', name + '_shape2'], [name + '_reshape2_out']), make_node('Mul', [name + '_reshape2_out', name + '_1_over_sqrt_e'], [name + '_mul0_out']), make_node('Slice', [name + '_reshape0_out', name + '_slice_start1', name + '_slice_end1'], [name + '_slice1_out']), make_node('Reshape', [name + '_slice1_out', name + '_shape1'], [name + '_reshape3_out']), make_node('Transpose', [name + '_reshape3_out'], [name + '_transpose1_out'], perm=(1, 2, 0, 3)), make_node('Reshape', [name + '_transpose1_out', name + '_shape2'], [name + '_reshape4_out']), make_node('Transpose', [name + '_reshape4_out'], [name + '_transpose2_out'], perm=(0, 2, 1)), make_node('MatMul', [name + '_mul0_out', name + '_transpose2_out'], [name], name=name)]
    return nodes