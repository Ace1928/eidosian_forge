import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('sin')
def convert_sin(node, **kwargs):
    """Map MXNet's sin operator attributes to onnx's Sin operator
    and return the created node.
    """
    return create_basic_op_node('Sin', node, kwargs)