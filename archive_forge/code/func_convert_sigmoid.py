import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('sigmoid')
def convert_sigmoid(node, **kwargs):
    """Map MXNet's sigmoid operator attributes to onnx's Sigmoid operator
    and return the created node.
    """
    return create_basic_op_node('Sigmoid', node, kwargs)