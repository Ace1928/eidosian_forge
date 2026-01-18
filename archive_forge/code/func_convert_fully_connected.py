import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('FullyConnected')
def convert_fully_connected(node, **kwargs):
    """Map MXNet's FullyConnected operator attributes to onnx's Gemm operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    flatten = get_boolean_attribute_value(attrs, 'flatten')
    no_bias = get_boolean_attribute_value(attrs, 'no_bias')
    num_hidden = int(attrs.get('num_hidden'))
    nodes = []
    if flatten:
        nodes += [make_node('Flatten', [input_nodes[0]], [name + '_data_flattened'])]
    else:
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_orig_shape']), make_node('Shape', [name + '_orig_shape'], [name + '_dim']), make_node('Flatten', [input_nodes[0]], [name + '_data_flattened'], axis=-1)]
    in_nodes = [name + '_data_flattened', input_nodes[1]]
    if no_bias:
        create_const_scalar_node(name + '_bias', np.int32(0).astype(dtype), kwargs)
        in_nodes.append(name + '_bias')
    else:
        in_nodes.append(input_nodes[2])
    if flatten:
        nodes += [make_node('Gemm', in_nodes, [name], alpha=1.0, beta=1.0, transA=0, transB=1, name=name)]
    else:
        create_tensor([0], name + '_0', kwargs['initializer'])
        create_tensor([1], name + '_1', kwargs['initializer'])
        create_tensor([num_hidden], name + '_num_hidden', kwargs['initializer'])
        nodes += [make_node('Gemm', in_nodes, [name + '_gemm'], alpha=1.0, beta=1.0, transA=0, transB=1), make_node('Sub', [name + '_dim', name + '_1'], [name + 'dim_minus_1']), make_node('Slice', [name + '_orig_shape', name + '_0', name + 'dim_minus_1'], [name + '_shape_sliced']), make_node('Concat', [name + '_shape_sliced', name + '_num_hidden'], [name + '_shape_new'], axis=0), make_node('Reshape', [name + '_gemm', name + '_shape_new'], [name], name=name)]
    return nodes