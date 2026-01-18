import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_minimum')
def convert_minimum(node, **kwargs):
    """Map MXNet's _minimum operator attributes to onnx's Min operator
    and return the created node.
    """
    return create_basic_op_node('Min', node, kwargs)