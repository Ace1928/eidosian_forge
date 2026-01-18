import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('depth_to_space')
def convert_depthtospace(node, **kwargs):
    """Map MXNet's depth_to_space operator attributes to onnx's
    DepthToSpace operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    blksize = int(attrs.get('block_size', 0))
    node = onnx.helper.make_node('DepthToSpace', input_nodes, [name], blocksize=blksize, name=name)
    return [node]