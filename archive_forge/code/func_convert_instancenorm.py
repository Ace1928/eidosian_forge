import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('InstanceNorm')
def convert_instancenorm(node, **kwargs):
    """Map MXNet's InstanceNorm operator attributes to onnx's InstanceNormalization operator
    based on the input node's attributes and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    eps = float(attrs.get('eps', 0.001))
    node = onnx.helper.make_node('InstanceNormalization', inputs=input_nodes, outputs=[name], name=name, epsilon=eps)
    return [node]