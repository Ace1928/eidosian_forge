import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_split_v2')
def convert_contrib_split_v2(node, **kwargs):
    """Map MXNet's _split_v2 operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))
    squeeze_axis = attrs.get('squeeze_axis', 'False')
    sections = int(attrs.get('sections', 0))
    indices = convert_string_to_list(attrs.get('indices', '[]'))
    if sections <= 0 and len(indices) == 0:
        raise NotImplementedError('section or indices must be set')
    if sections > 0:
        output_nodes = [name + str(i) for i in range(sections)]
        if squeeze_axis == 'False':
            nodes = [make_node('Split', input_nodes, output_nodes, axis=axis)]
        else:
            output_nodes_ = [name + str(i) + '_' for i in range(sections)]
            nodes = [make_node('Split', input_nodes, output_nodes_, axis=axis)]
            for i in range(sections):
                nodes += [make_node('Squeeze', [output_nodes_[i]], [output_nodes[i]], axes=[axis])]
    else:
        raise NotImplementedError('indices is supported since ONNX 1.8.0 (opset13), please upgrade ONNX version')
    return nodes