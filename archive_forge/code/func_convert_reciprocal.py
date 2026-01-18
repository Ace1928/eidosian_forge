import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('reciprocal')
def convert_reciprocal(node, **kwargs):
    """Map MXNet's reciprocal operator attributes to onnx's Reciprocal operator
    and return the created node.
    """
    return create_basic_op_node('Reciprocal', node, kwargs)