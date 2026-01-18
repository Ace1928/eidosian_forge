import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('L2Normalization')
def convert_l2normalization(node, **kwargs):
    """Map MXNet's L2Normalization operator attributes to onnx's LpNormalization operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    mode = attrs.get('mode', 'instance')
    if mode != 'channel':
        raise AttributeError('L2Normalization: ONNX currently supports channel mode only')
    l2norm_node = onnx.helper.make_node('LpNormalization', input_nodes, [name], axis=1, name=name)
    return [l2norm_node]