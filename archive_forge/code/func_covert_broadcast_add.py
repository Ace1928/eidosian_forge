import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_add')
def covert_broadcast_add(node, **kwargs):
    """Map MXNet's broadcast_add operator attributes to onnx's Add operator
    and return the created node.
    """
    return create_basic_op_node('Add', node, kwargs)