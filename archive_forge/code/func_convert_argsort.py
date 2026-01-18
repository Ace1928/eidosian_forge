import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('argsort')
def convert_argsort(node, **kwargs):
    """Map MXNet's argsort operator attributes to onnx's TopK operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')
    axis = int(attrs.get('axis', '-1'))
    is_ascend = attrs.get('is_ascend', 'True')
    is_ascend = is_ascend in ['True', '1']
    dtype = attrs.get('dtype', 'float32')
    create_tensor([axis], name + '_axis', kwargs['initializer'])
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Gather', [name + '_shape', name + '_axis'], [name + '_k'])]
    if dtype == 'int64':
        nodes += [make_node('TopK', [input_nodes[0], name + '_k'], [name + '_', name], axis=axis, largest=not is_ascend, sorted=1)]
    else:
        nodes += [make_node('TopK', [input_nodes[0], name + '_k'], [name + '_', name + '_temp'], axis=axis, largest=not is_ascend, sorted=1), make_node('Cast', [name + '_temp'], [name], to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)])]
    return nodes