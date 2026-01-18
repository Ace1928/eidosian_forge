import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_rdiv_scalar')
def convert_rdiv_scalar(node, **kwargs):
    """Map MXNet's _rdiv_scalar operator attributes to onnx's Div operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Div', **kwargs)