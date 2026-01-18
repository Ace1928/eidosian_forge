import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('norm')
def convert_norm(node, **kwargs):
    """Map MXNet's norm operator attributes to onnx's ReduceL1 and ReduceL2 operators
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    mx_axis = attrs.get('axis', None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis else None
    keepdims = get_boolean_attribute_value(attrs, 'keepdims')
    ord = int(attrs.get('ord', 2))
    if ord not in [1, 2]:
        raise AttributeError('norm export operator only supports ord=1 or ord=2.')
    onnx_op_name = 'ReduceL1' if ord == 1 else 'ReduceL2'
    if axes:
        if keepdims:
            reduce_node = make_node(onnx_op_name, input_nodes, [name], axes=axes, keepdims=keepdims)
            return [reduce_node]
        else:
            create_tensor([1], name + '_1', kwargs['initializer'])
            nodes = [make_node(onnx_op_name, input_nodes, [name + '_norm'], axes=axes, keepdims=keepdims), make_node('Shape', [name + '_norm'], [name + '_norm_shape']), make_node('Concat', [name + '_1', name + '_norm_shape'], [name + '_concat'], axis=0), make_node('Reshape', [name + '_norm', name + '_concat'], [name + '_reshape']), make_node('Squeeze', [name + '_reshape'], [name], axes=[0])]
            return nodes
    elif keepdims:
        reduce_node = make_node(onnx_op_name, input_nodes, [name], keepdims=keepdims)
        return [reduce_node]
    else:
        create_tensor([1], name + '_1', kwargs['initializer'])
        nodes = [make_node(onnx_op_name, input_nodes, [name + '_norm'], keepdims=keepdims), make_node('Reshape', [name + '_norm', name + '_1'], [name])]
        return nodes