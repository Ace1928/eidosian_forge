import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_rnn_param_concat')
def convert_rnn_param_concat(node, **kwargs):
    """Map MXNet's _rnn_param_concat operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('dim', 1))
    nodes = [make_node('Concat', input_nodes, [name], axis=axis)]
    return nodes