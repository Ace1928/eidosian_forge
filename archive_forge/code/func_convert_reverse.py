import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('reverse')
def convert_reverse(node, **kwargs):
    """Map MXNet's reverse operator attributes to ONNX
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))
    perm = [i for i in range(8)]
    perm[0], perm[axis] = (axis, 0)
    create_tensor([8], name + '_8', kwargs['initializer'])
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([-1], name + '_m1', kwargs['initializer'])
    create_tensor([axis], name + '_axis', kwargs['initializer'])
    create_tensor([axis + 1], name + '_axis_p1', kwargs['initializer'])
    create_const_scalar_node(name + '_m1_s', np.int64(-1), kwargs)
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Shape', [name + '_shape'], [name + '_dim']), make_node('Sub', [name + '_8', name + '_dim'], [name + '_sub']), make_node('Concat', [name + '_0', name + '_sub'], [name + '_concat'], axis=0), make_node('Pad', [name + '_shape', name + '_concat', name + '_1'], [name + '_shape_8_dim']), make_node('Reshape', [input_nodes[0], name + '_shape_8_dim'], [name + '_data_8_dim']), make_node('Transpose', [name + '_data_8_dim'], [name + '_data_t'], perm=perm), make_node('Slice', [name + '_shape', name + '_axis', name + '_axis_p1'], [name + '_axis_len']), make_node('Sub', [name + '_axis_len', name + '_1'], [name + '_axis_len_m1']), make_node('Squeeze', [name + '_axis_len_m1'], [name + '_axis_len_m1_s'], axes=[0]), make_node('Range', [name + '_axis_len_m1_s', name + '_m1_s', name + '_m1_s'], [name + '_indices']), make_node('Gather', [name + '_data_t', name + '_indices'], [name + '_gather']), make_node('Transpose', [name + '_gather'], [name + '_data_reversed'], perm=perm), make_node('Reshape', [name + '_data_reversed', name + '_shape'], [name], name=name)]
    return nodes