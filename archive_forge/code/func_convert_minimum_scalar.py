import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_minimum_scalar')
def convert_minimum_scalar(node, **kwargs):
    """Map MXNet's _minimum_scalar
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    scalar = None
    if 'float' in str(dtype):
        scalar = float(attrs.get('scalar', '0'))
    else:
        scalar = int(attrs.get('scalar', '0'))
    create_tensor([scalar], name + '_scalar', kwargs['initializer'], dtype=dtype)
    nodes = [make_node('Min', [input_nodes[0], name + '_scalar'], [name], name=name)]
    return nodes