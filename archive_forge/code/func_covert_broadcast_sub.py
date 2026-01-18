import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_sub')
def covert_broadcast_sub(node, **kwargs):
    """Map MXNet's broadcast_sub operator attributes to onnx's Sub operator
    and return the created node.
    """
    return create_basic_op_node('Sub', node, kwargs)