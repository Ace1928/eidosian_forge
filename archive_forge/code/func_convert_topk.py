import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('topk')
def convert_topk(node, **kwargs):
    """Map MXNet's topk operator attributes to onnx's TopK operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')
    axis = int(attrs.get('axis', '-1'))
    k = int(attrs.get('k', '1'))
    ret_type = attrs.get('ret_typ', 'indices')
    is_ascend = attrs.get('is_ascend', 'False')
    is_ascend = is_ascend in ['1', 'True']
    dtype = attrs.get('dtype', 'float32')
    if ret_type == 'mask':
        raise NotImplementedError("topk does not currently support ret_type=='mask'")
    create_tensor([k], name + '_k', kwargs['initializer'])
    nodes = []
    if ret_type == 'both':
        if dtype == 'int64':
            nodes += [make_node('TopK', [input_nodes[0], name + '_k'], [name + '0', name + '1'], axis=axis, largest=not is_ascend, sorted=1)]
        else:
            nodes += [make_node('TopK', [input_nodes[0], name + '_k'], [name + '0', name + '_1_i'], axis=axis, largest=not is_ascend, sorted=1), make_node('Cast', [name + '_1_i'], [name + '1'], to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)])]
    elif ret_type == 'value':
        nodes += [make_node('TopK', [input_nodes[0], name + '_k'], [name + '0', name + '_'], axis=axis, largest=not is_ascend, sorted=1)]
    elif dtype == 'int64':
        nodes += [make_node('TopK', [input_nodes[0], name + '_k'], [name + '_', name], axis=axis, largest=not is_ascend, sorted=1)]
    else:
        nodes += [make_node('TopK', [input_nodes[0], name + '_k'], [name + '__', name + '_tmp'], axis=axis, largest=not is_ascend, sorted=1), make_node('Cast', [name + '_tmp'], [name], to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)])]
    return nodes