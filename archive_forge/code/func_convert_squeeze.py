import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('squeeze')
def convert_squeeze(node, **kwargs):
    """Map MXNet's squeeze operator attributes to onnx's squeeze operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    mx_axis = str(attrs.get('axis', 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None
    if not axes:
        node = onnx.helper.make_node('Squeeze', input_nodes, [name], name=name)
    else:
        node = onnx.helper.make_node('Squeeze', input_nodes, [name], axes=axes, name=name)
    return [node]