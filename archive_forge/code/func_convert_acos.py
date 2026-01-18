import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('arccos')
def convert_acos(node, **kwargs):
    """Map MXNet's acos operator attributes to onnx's acos operator
    and return the created node.
    """
    return create_basic_op_node('Acos', node, kwargs)