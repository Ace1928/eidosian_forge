import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('transpose')
def convert_transpose(node, **kwargs):
    """Map MXNet's transpose operator attributes to onnx's Transpose operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axes = attrs.get('axes', ())
    if axes:
        axes = tuple(map(int, re.findall('\\d+', axes)))
        transpose_node = onnx.helper.make_node('Transpose', input_nodes, [name], perm=axes, name=name)
    else:
        transpose_node = onnx.helper.make_node('Transpose', input_nodes, [name], name=name)
    return [transpose_node]