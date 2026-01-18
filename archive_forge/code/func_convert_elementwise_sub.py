import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('elemwise_sub')
def convert_elementwise_sub(node, **kwargs):
    """Map MXNet's elemwise_sub operator attributes to onnx's Sub operator
    and return the created node.
    """
    return create_basic_op_node('Sub', node, kwargs)