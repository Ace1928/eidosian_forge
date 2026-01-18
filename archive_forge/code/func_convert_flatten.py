import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Flatten')
def convert_flatten(node, **kwargs):
    """Map MXNet's Flatten operator attributes to onnx's Flatten operator
    and return the created node.
    """
    return create_basic_op_node('Flatten', node, kwargs)