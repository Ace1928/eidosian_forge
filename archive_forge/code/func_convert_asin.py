import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('arcsin')
def convert_asin(node, **kwargs):
    """Map MXNet's asin operator attributes to onnx's asin operator
    and return the created node.
    """
    return create_basic_op_node('Asin', node, kwargs)