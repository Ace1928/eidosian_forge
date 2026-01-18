import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('SliceChannel')
def convert_slice_channel(node, **kwargs):
    """Map MXNet's SliceChannel operator attributes to onnx's Squeeze or Split
    operator based on squeeze_axis attribute
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    num_outputs = int(attrs.get('num_outputs'))
    axis = int(attrs.get('axis', 1))
    squeeze_axis = attrs.get('squeeze_axis', 'False')
    nodes = []
    if squeeze_axis in ['True', '1']:
        nodes += [make_node('Split', [input_nodes[0]], [name + str(i) + '_' for i in range(num_outputs)], axis=axis)]
        for i in range(num_outputs):
            nodes += [make_node('Squeeze', [name + str(i) + '_'], [name + str(i)], axes=[axis])]
    else:
        nodes += [make_node('Split', [input_nodes[0]], [name + str(i) for i in range(num_outputs)], axis=axis)]
    return nodes