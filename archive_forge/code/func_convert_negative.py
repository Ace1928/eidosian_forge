import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('negative')
def convert_negative(node, **kwargs):
    """Map MXNet's negative operator attributes to onnx's Neg operator
    and return the created node.
    """
    return create_basic_op_node('Neg', node, kwargs)