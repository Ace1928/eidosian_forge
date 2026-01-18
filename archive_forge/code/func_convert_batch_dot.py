import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('batch_dot')
def convert_batch_dot(node, **kwargs):
    """Map MXNet's batch_dot operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    lhs = input_nodes[0]
    rhs = input_nodes[1]
    transpose_a = str(attrs.get('transpose_a', 'False'))
    transpose_b = str(attrs.get('transpose_b', 'False'))
    perm = [0, 2, 1]
    if transpose_a == 'False' and transpose_b == 'False':
        nodes = [make_node('MatMul', [lhs, rhs], [name])]
        return nodes
    create_tensor([-2], name + '_-2', kwargs['initializer'])
    create_tensor([-1], name + '_-1', kwargs['initializer'])
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([100], name + '_100', kwargs['initializer'])
    nodes = []
    if transpose_a != 'False' and transpose_b == 'False':
        nodes += [make_node('Shape', [lhs], [name + '_lhs_shape']), make_node('Shape', [name + '_lhs_shape'], [name + '_lhs_dim']), make_node('Slice', [name + '_lhs_shape', name + '_0', name + '_-2'], [name + '_lhs_slice0']), make_node('Slice', [name + '_lhs_shape', name + '_-2', name + '_100'], [name + '_lhs_slice1']), make_node('Concat', [name + '_-1', name + '_lhs_slice1'], [name + '_lhs_concat1'], axis=0), make_node('Reshape', [lhs, name + '_lhs_concat1'], [name + '_lhs_3d']), make_node('Transpose', [name + '_lhs_3d'], [name + '_lhs_3d_transpose'], perm=perm), make_node('Shape', [name + '_lhs_3d_transpose'], [name + '_lhs_shape_3d']), make_node('Slice', [name + '_lhs_shape_3d', name + '_-2', name + '_100'], [name + '_lhs_slice2']), make_node('Concat', [name + '_lhs_slice0', name + '_lhs_slice2'], [name + '_lhs_concat2'], axis=0), make_node('Reshape', [name + '_lhs_3d_transpose', name + '_lhs_concat2'], [name + '_lhs']), make_node('MatMul', [name + '_lhs', rhs], [name])]
    elif transpose_a == 'False' and transpose_b != 'False':
        nodes += [make_node('Shape', [rhs], [name + '_rhs_shape']), make_node('Shape', [name + '_rhs_shape'], [name + '_rhs_dim']), make_node('Slice', [name + '_rhs_shape', name + '_0', name + '_-2'], [name + '_rhs_slice0']), make_node('Slice', [name + '_rhs_shape', name + '_-2', name + '_100'], [name + '_rhs_slice1']), make_node('Concat', [name + '_-1', name + '_rhs_slice1'], [name + '_rhs_concat1'], axis=0), make_node('Reshape', [rhs, name + '_rhs_concat1'], [name + '_rhs_3d']), make_node('Transpose', [name + '_rhs_3d'], [name + '_rhs_3d_transpose'], perm=perm), make_node('Shape', [name + '_rhs_3d_transpose'], [name + '_rhs_shape_3d']), make_node('Slice', [name + '_rhs_shape_3d', name + '_-2', name + '_100'], [name + '_rhs_slice2']), make_node('Concat', [name + '_rhs_slice0', name + '_rhs_slice2'], [name + '_rhs_concat2'], axis=0), make_node('Reshape', [name + '_rhs_3d_transpose', name + '_rhs_concat2'], [name + '_rhs']), make_node('MatMul', [lhs, name + '_rhs'], [name])]
    else:
        nodes += [make_node('Shape', [lhs], [name + '_lhs_shape']), make_node('Shape', [name + '_lhs_shape'], [name + '_lhs_dim']), make_node('Slice', [name + '_lhs_shape', name + '_0', name + '_-2'], [name + '_lhs_slice0']), make_node('Slice', [name + '_lhs_shape', name + '_-2', name + '_100'], [name + '_lhs_slice1']), make_node('Concat', [name + '_-1', name + '_lhs_slice1'], [name + '_lhs_concat1'], axis=0), make_node('Reshape', [lhs, name + '_lhs_concat1'], [name + '_lhs_3d']), make_node('Transpose', [name + '_lhs_3d'], [name + '_lhs_3d_transpose'], perm=perm), make_node('Shape', [name + '_lhs_3d_transpose'], [name + '_lhs_shape_3d']), make_node('Slice', [name + '_lhs_shape_3d', name + '_-2', name + '_100'], [name + '_lhs_slice2']), make_node('Concat', [name + '_lhs_slice0', name + '_lhs_slice2'], [name + '_lhs_concat2'], axis=0), make_node('Reshape', [name + '_lhs_3d_transpose', name + '_lhs_concat2'], [name + '_lhs']), make_node('Shape', [rhs], [name + '_rhs_shape']), make_node('Shape', [name + '_rhs_shape'], [name + '_rhs_dim']), make_node('Slice', [name + '_rhs_shape', name + '_0', name + '_-2'], [name + '_rhs_slice0']), make_node('Slice', [name + '_rhs_shape', name + '_-2', name + '_100'], [name + '_rhs_slice1']), make_node('Concat', [name + '_-1', name + '_rhs_slice1'], [name + '_rhs_concat1'], axis=0), make_node('Reshape', [rhs, name + '_rhs_concat1'], [name + '_rhs_3d']), make_node('Transpose', [name + '_rhs_3d'], [name + '_rhs_3d_transpose'], perm=perm), make_node('Shape', [name + '_rhs_3d_transpose'], [name + '_rhs_shape_3d']), make_node('Slice', [name + '_rhs_shape_3d', name + '_-2', name + '_100'], [name + '_rhs_slice2']), make_node('Concat', [name + '_rhs_slice0', name + '_rhs_slice2'], [name + '_rhs_concat2'], axis=0), make_node('Reshape', [name + '_rhs_3d_transpose', name + '_rhs_concat2'], [name + '_rhs']), make_node('MatMul', [name + '_lhs', name + '_rhs'], [name])]
    return nodes