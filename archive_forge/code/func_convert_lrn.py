import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('LRN')
def convert_lrn(node, **kwargs):
    """Map MXNet's LRN operator attributes to onnx's LRN operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    alpha = float(attrs.get('alpha', 0.0001))
    beta = float(attrs.get('beta', 0.75))
    bias = float(attrs.get('knorm', 1.0))
    size = int(attrs.get('nsize'))
    lrn_node = onnx.helper.make_node('LRN', inputs=input_nodes, outputs=[name], name=name, alpha=alpha, beta=beta, bias=bias, size=size)
    return [lrn_node]