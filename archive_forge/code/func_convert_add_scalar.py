import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_plus_scalar')
def convert_add_scalar(node, **kwargs):
    """Map MXNet's _plus_scalar operator attributes to onnx's Add operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Add', **kwargs)