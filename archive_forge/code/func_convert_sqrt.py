import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('sqrt')
def convert_sqrt(node, **kwargs):
    """Map MXNet's sqrt operator attributes to onnx's Sqrt operator
    and return the created node.
    """
    return create_basic_op_node('Sqrt', node, kwargs)