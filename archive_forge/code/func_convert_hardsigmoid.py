import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('hard_sigmoid')
def convert_hardsigmoid(node, **kwargs):
    """Map MXNet's hard_sigmoid operator attributes to onnx's HardSigmoid operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    alpha = float(attrs.get('alpha', 0.2))
    beta = float(attrs.get('beta', 0.5))
    node = onnx.helper.make_node('HardSigmoid', input_nodes, [name], alpha=alpha, beta=beta, name=name)
    return [node]