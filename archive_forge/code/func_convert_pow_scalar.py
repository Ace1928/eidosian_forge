import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_power_scalar')
def convert_pow_scalar(node, **kwargs):
    """Map MXNet's _pow_scalar operator attributes to onnx's Pow operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Pow', **kwargs)