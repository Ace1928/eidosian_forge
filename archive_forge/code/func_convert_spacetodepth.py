import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('space_to_depth')
def convert_spacetodepth(node, **kwargs):
    """Map MXNet's space_to_depth operator attributes to onnx's
    SpaceToDepth operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    blksize = int(attrs.get('block_size', 0))
    node = onnx.helper.make_node('SpaceToDepth', input_nodes, [name], blocksize=blksize, name=name)
    return [node]