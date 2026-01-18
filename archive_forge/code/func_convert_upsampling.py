import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('UpSampling')
def convert_upsampling(node, **kwargs):
    """Map MXNet's UpSampling operator to onnx.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    scale = int(attrs.get('scale', '1'))
    sample_type = attrs.get('sample_type')
    num_args = int(attrs.get('num_args', '1'))
    if num_args > 1:
        raise NotImplementedError('Upsampling conversion does not currently support num_args > 1')
    if sample_type != 'nearest':
        raise NotImplementedError('Upsampling conversion does not currently support                                    sample_type != nearest')
    create_tensor([], name + '_roi', kwargs['initializer'], dtype='float32')
    create_tensor([1, 1, scale, scale], name + '_scales', kwargs['initializer'], dtype='float32')
    nodes = [make_node('Resize', [input_nodes[0], name + '_roi', name + '_scales'], [name], mode='nearest', coordinate_transformation_mode='half_pixel')]
    return nodes