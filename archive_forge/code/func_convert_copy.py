import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_copy')
def convert_copy(node, **kwargs):
    """Map MXNet's _copy operator attributes to onnx's Identity operator
    and return the created node.
    """
    return create_basic_op_node('Identity', node, kwargs)