import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('log')
def convert_log(node, **kwargs):
    """Map MXNet's log operator attributes to onnx's Log operator
    and return the created node.
    """
    return create_basic_op_node('Log', node, kwargs)