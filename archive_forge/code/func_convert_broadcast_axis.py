import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_axis')
def convert_broadcast_axis(node, **kwargs):
    """Map MXNet's broadcast_axis
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = convert_string_to_list(attrs.get('axis', '()'))
    size = convert_string_to_list(attrs.get('size', '()'))
    assert len(axis) == len(size)
    shape_name = name + '_shape_0'
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_const_scalar_node(name + '_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name + '_1_s', np.int64(1), kwargs)
    nodes = [make_node('Shape', [input_nodes[0]], [shape_name]), make_node('Shape', [shape_name], [name + '_in_dim']), make_node('Squeeze', [name + '_in_dim'], [name + '_in_dim_s'], axes=[0]), make_node('Range', [name + '_0_s', name + '_in_dim_s', name + '_1_s'], [name + '_range'])]
    for i, axis in enumerate(axis):
        if axis not in (0, 1):
            create_tensor([axis], name + '_' + str(axis), kwargs['initializer'])
        create_tensor([size[i] - 1], name + '_size_' + str(i), kwargs['initializer'])
        nodes += [make_node('Equal', [name + '_range', name + '_' + str(axis)], [name + '_equal_' + str(i)]), make_node('Cast', [name + '_equal_' + str(i)], [name + '_cast_' + str(i)], to=int(TensorProto.INT64)), make_node('Mul', [name + '_size_' + str(i), name + '_cast_' + str(i)], [name + '_mul_' + str(i)]), make_node('Add', [name + '_mul_' + str(i), name + '_1'], [name + '_add_' + str(i)]), make_node('Mul', [name + '_add_' + str(i), shape_name], [name + '_shape_' + str(i + 1)])]
        shape_name = name + '_shape_' + str(i + 1)
    nodes += [make_node('Expand', [input_nodes[0], shape_name], [name], name=name)]
    return nodes