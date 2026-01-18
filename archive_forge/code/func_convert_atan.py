import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('arctan')
def convert_atan(node, **kwargs):
    """Map MXNet's atan operator attributes to onnx's atan operator
    and return the created node.
    """
    return create_basic_op_node('Atan', node, kwargs)