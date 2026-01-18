import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
def create_helper_trans_node(node_name, input_node):
    """create extra transpose node for dot operator"""
    trans_node = onnx.helper.make_node('Transpose', inputs=[input_node], outputs=[node_name], name=node_name)
    return trans_node