import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('abs')
def convert_abs(node, **kwargs):
    """Map MXNet's abs operator attributes to onnx's Abs operator
    and return the created node.
    """
    return create_basic_op_node('Abs', node, kwargs)