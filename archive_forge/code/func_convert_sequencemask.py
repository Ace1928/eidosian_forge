import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('SequenceMask')
def convert_sequencemask(node, **kwargs):
    """Map MXNet's SequenceMask operator
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    use_sequence_length = attrs.get('use_sequence_length', 'False')
    mask_val = float(attrs.get('value', '0'))
    axis = int(attrs.get('axis', '0'))
    if use_sequence_length == 'False':
        return [make_node('Identity', [input_nodes[0]], [name], name=name)]
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([2], name + '_2', kwargs['initializer'])
    create_const_scalar_node(name + '_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name + '_1_s', np.int64(1), kwargs)
    create_const_scalar_node(name + '_2_s', np.int64(2), kwargs)
    create_tensor([mask_val], name + '_mask_val', kwargs['initializer'], dtype='float32')
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_in_shape']), make_node('Slice', [name + '_in_shape', name + '_0', name + '_1'], [name + '_slice_0']), make_node('Slice', [name + '_in_shape', name + '_1', name + '_2'], [name + '_slice_1']), make_node('Concat', [name + '_slice_0', name + '_1'], [name + '_shape_0'], axis=0), make_node('Shape', [name + '_in_shape'], [name + '_in_dim']), make_node('Squeeze', [name + '_in_dim'], [name + '_in_dim_s'], axes=[0]), make_node('Range', [name + '_0_s', name + '_in_dim_s', name + '_1_s'], [name + '_range_0']), make_node('Less', [name + '_range_0', name + '_2'], [name + '_less_0']), make_node('Where', [name + '_less_0', name + '_in_shape', name + '_1'], [name + '_shape_1'])]
    if axis == 0:
        nodes += [make_node('Squeeze', [name + '_slice_0'], [name + '_max_len'], axes=[0]), make_node('Range', [name + '_0_s', name + '_max_len', name + '_1_s'], [name + '_range_1']), make_node('Reshape', [name + '_range_1', name + '_shape_0'], [name + '_reshape_0']), make_node('Cast', [input_nodes[1]], [name + '_cast'], to=int(TensorProto.INT64)), make_node('Less', [name + '_reshape_0', name + '_cast'], [name + '_less_1']), make_node('Reshape', [name + '_less_1', name + '_shape_1'], [name + '_reshape_1']), make_node('Where', [name + '_reshape_1', input_nodes[0], name + '_mask_val'], [name], name=name)]
    else:
        nodes += [make_node('Squeeze', [name + '_slice_1'], [name + '_max_len'], axes=[0]), make_node('Range', [name + '_0_s', name + '_max_len', name + '_1_s'], [name + '_range_1']), make_node('Reshape', [input_nodes[1], name + '_shape_0'], [name + '_reshape_0']), make_node('Cast', [name + '_reshape_0'], [name + '_cast'], to=int(TensorProto.INT64)), make_node('Less', [name + '_range_1', name + '_cast'], [name + '_less_1']), make_node('Reshape', [name + '_less_1', name + '_shape_1'], [name + '_reshape_1']), make_node('Where', [name + '_reshape_1', input_nodes[0], name + '_mask_val'], [name], name=name)]
    return nodes