import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('BatchNorm')
def convert_batchnorm(node, **kwargs):
    """Map MXNet's BatchNorm operator attributes to onnx's BatchNormalization operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    momentum = float(attrs.get('momentum', 0.9))
    eps = float(attrs.get('eps', 0.001))
    axis = int(attrs.get('axis', 1))
    if axis != 1:
        raise NotImplementedError('batchnorm axis != 1 is currently not supported.')
    bn_node = onnx.helper.make_node('BatchNormalization', input_nodes, [name], name=name, epsilon=eps, momentum=momentum)
    return [bn_node]