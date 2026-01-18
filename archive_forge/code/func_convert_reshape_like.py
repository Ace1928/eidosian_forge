import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('reshape_like')
def convert_reshape_like(node, **kwargs):
    """Map MXNet's reshape_like operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    lhs = input_nodes[0]
    rhs = input_nodes[1]
    lhs_begin = str(attrs.get('lhs_begin', '0'))
    rhs_begin = str(attrs.get('rhs_begin', '0'))
    lhs_end = str(attrs.get('lhs_end', 'None'))
    rhs_end = str(attrs.get('rhs_end', 'None'))
    if lhs_begin == 'None' or rhs_begin == 'None':
        raise NotImplementedError('lhs_begin and rhs_begin should not be None.')
    lhs_begin = int(lhs_begin)
    rhs_begin = int(rhs_begin)
    if lhs_begin == 0 and lhs_end == 'None' and (rhs_begin == 0) and (rhs_end == 'None'):
        nodes = [make_node('Shape', [rhs], [name + '_shape_rhs']), make_node('Reshape', [lhs, name + '_shape_rhs'], [name], name=name)]
        return nodes
    create_tensor([0], name + '_0', kwargs['initializer'])
    nodes = [make_node('Shape', [lhs], [name + '_lhs_shape']), make_node('Shape', [name + '_lhs_shape'], [name + '_lhs_dim']), make_node('Shape', [rhs], [name + '_rhs_shape']), make_node('Shape', [name + '_rhs_shape'], [name + '_rhs_dim'])]
    if lhs_begin >= 0:
        create_tensor([lhs_begin], name + '_lhs_begin', kwargs['initializer'])
    else:
        create_tensor([lhs_begin], name + '_lhs_begin_neg', kwargs['initializer'])
        nodes += [make_node('Add', [name + '_lhs_dim', name + '_lhs_begin_neg'], [name + '_lhs_begin'])]
    if rhs_begin >= 0:
        create_tensor([rhs_begin], name + '_rhs_begin', kwargs['initializer'])
    else:
        create_tensor([rhs_begin], name + '_rhs_begin_neg', kwargs['initializer'])
        nodes += [make_node('Add', [name + '_rhs_dim', name + '_rhs_begin_neg'], [name + '_rhs_begin'])]
    if lhs_end == 'None':
        nodes += [make_node('Add', [name + '_lhs_dim', name + '_0'], [name + '_lhs_end'])]
    else:
        lhs_end = int(lhs_end)
        if lhs_end >= 0:
            create_tensor([lhs_end], name + '_lhs_end', kwargs['initializer'])
        else:
            create_tensor([lhs_end], name + '_lhs_end_neg', kwargs['initializer'])
            nodes += [make_node('Add', [name + '_lhs_dim', name + '_lhs_end_neg'], [name + '_lhs_end'])]
    if rhs_end == 'None':
        nodes += [make_node('Add', [name + '_rhs_dim', name + '_0'], [name + '_rhs_end'])]
    else:
        rhs_end = int(rhs_end)
        if rhs_end >= 0:
            create_tensor([rhs_end], name + '_rhs_end', kwargs['initializer'])
        else:
            create_tensor([rhs_end], name + '_rhs_end_neg', kwargs['initializer'])
            nodes += [make_node('Add', [name + '_rhs_dim', name + '_rhs_end_neg'], [name + '_rhs_end'])]
    nodes += [make_node('Slice', [name + '_lhs_shape', name + '_0', name + '_lhs_begin'], [name + '_slice0_out']), make_node('Slice', [name + '_rhs_shape', name + '_rhs_begin', name + '_rhs_end'], [name + '_slice1_out']), make_node('Concat', [name + '_slice0_out', name + '_slice1_out'], [name + '_concat0_out'], axis=0), make_node('Slice', [name + '_lhs_shape', name + '_lhs_end', name + '_lhs_dim'], [name + '_slice2_out']), make_node('Concat', [name + '_concat0_out', name + '_slice2_out'], [name + '_concat1_out'], axis=0), make_node('Reshape', [lhs, name + '_concat1_out'], [name], name=name)]
    return nodes