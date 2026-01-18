import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_maximum')
def convert_maximum(node, **kwargs):
    """Map MXNet's _maximum operator attributes to onnx's Max operator
    and return the created node.
    """
    return create_basic_op_node('Max', node, kwargs)