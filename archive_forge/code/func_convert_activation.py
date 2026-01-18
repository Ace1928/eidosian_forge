import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Activation')
def convert_activation(node, **kwargs):
    """Map MXNet's Activation operator attributes to onnx's Tanh/Relu operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    act_type = attrs['act_type']
    act_types = {'tanh': 'Tanh', 'relu': 'Relu', 'sigmoid': 'Sigmoid', 'softrelu': 'Softplus', 'softsign': 'Softsign'}
    act_name = act_types.get(act_type)
    if act_name:
        node = onnx.helper.make_node(act_name, input_nodes, [name], name=name)
    else:
        raise AttributeError('Activation %s not implemented or recognized in the converter' % act_type)
    return [node]