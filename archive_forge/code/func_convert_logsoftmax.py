import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('log_softmax')
def convert_logsoftmax(node, **kwargs):
    """Map MXNet's log_softmax operator attributes to onnx's LogSoftMax operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', -1))
    temp = attrs.get('temperature', 'None')
    use_length = attrs.get('use_length', 'False')
    if temp != 'None':
        raise AttributeError('LogSoftMax currently does not support temperature!=None')
    if use_length in ['1', 'True']:
        raise AttributeError('LogSoftMax currently does not support use_length==True')
    nodes = [make_node('Exp', [input_nodes[0]], [name + '_exp']), make_node('ReduceSum', [name + '_exp'], [name + '_rsum'], axes=[axis], keepdims=1), make_node('Div', [name + '_exp', name + '_rsum'], [name + '_div']), make_node('Log', [name + '_div'], [name])]
    return nodes