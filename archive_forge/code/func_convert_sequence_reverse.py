import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('SequenceReverse')
def convert_sequence_reverse(node, **kwargs):
    """Map MXNet's SequenceReverse op
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    batch_axis = 1
    time_axis = 0
    use_sequence_length = attrs.get('use_sequence_length', 'False')
    nodes = []
    if use_sequence_length == 'False':
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Split', [name + '_shape'], [name + '_dim0', name + '_dim1', name + '_dim2']), make_node('Expand', [name + '_dim0', name + '_dim1'], [name + '_seq_len']), make_node('ReverseSequence', [input_nodes[0], name + '_seq_len'], [name], batch_axis=batch_axis, time_axis=time_axis)]
    else:
        nodes += [make_node('Cast', [input_nodes[1]], [name + '_seq_len'], to=int(TensorProto.INT64)), make_node('ReverseSequence', [input_nodes[0], name + '_seq_len'], [name], batch_axis=batch_axis, time_axis=time_axis)]
    return nodes