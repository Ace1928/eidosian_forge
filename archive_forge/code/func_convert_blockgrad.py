import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('BlockGrad')
def convert_blockgrad(node, **kwargs):
    """ Skip operator  """
    return create_basic_op_node('Identity', node, kwargs)