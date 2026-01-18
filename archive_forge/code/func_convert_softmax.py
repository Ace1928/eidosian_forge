import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('softmax')
def convert_softmax(node, **kwargs):
    """Map MXNet's softmax operator attributes to onnx's Softmax operator
    and return the created node.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    axis = int(attrs.get('axis', -1))
    temperature = str(attrs.get('temperature', 'None'))
    if temperature == 'None':
        temperature = 1.0
    else:
        temperature = float(temperature)
    use_length = str(attrs.get('use_length', 'None'))
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    data = input_nodes[0]
    if axis == -1 and temperature == 1.0:
        nodes = []
        if use_length == 'True':
            create_tensor([-65500.0], name + '_mask_val', kwargs['initializer'], dtype=dtype)
            create_tensor([1], name + '_1', kwargs['initializer'])
            create_const_scalar_node(name + '_0_s', np.int64(0), kwargs)
            create_const_scalar_node(name + '_1_s', np.int64(1), kwargs)
            nodes += [make_node('Shape', [data], [name + '_shape']), make_node('Shape', [name + '_shape'], [name + '_dim']), make_node('Sub', [name + '_dim', name + '_1'], [name + '_dim_m1']), make_node('Slice', [name + '_shape', name + '_dim_m1', name + '_dim'], [name + '_dim_last_']), make_node('Squeeze', [name + '_dim_last_'], [name + '_dim_last'], axes=[0]), make_node('Range', [name + '_0_s', name + '_dim_last', name + '_1_s'], [name + '_range']), make_node('Cast', [input_nodes[1]], [name + '_len'], to=int(TensorProto.INT64)), make_node('Unsqueeze', [name + '_len'], [name + '_len_unsqueezed'], axes=[-1]), make_node('Less', [name + '_range', name + '_len_unsqueezed'], [name + '_less']), make_node('Where', [name + '_less', data, name + '_mask_val'], [name + '_data_masked'])]
            data = name + '_data_masked'
        nodes += [make_node('Softmax', [data], [name], axis=-1)]
        return nodes
    create_tensor([temperature], name + '_tmp', kwargs['initializer'], dtype=dtype)
    nodes = [make_node('Div', [data, name + '_tmp'], [name + '_data']), make_node('Exp', [name + '_data'], [name + '_exp_out']), make_node('ReduceSum', [name + '_exp_out'], [name + '_rsum_out'], axes=[axis], keepdims=1)]
    if len(input_nodes) == 1:
        nodes += [make_node('Div', [name + '_exp_out', name + '_rsum_out'], [name], name=name)]
        return nodes
    elif use_length == 'True':
        length = input_nodes[1]
        create_tensor([axis], name + '_axis', kwargs['initializer'])
        create_tensor([0], name + '_0', kwargs['initializer'])
        create_tensor([1], name + '_1', kwargs['initializer'])
        create_const_scalar_node(name + '_-1_s', np.int64(-1), kwargs)
        create_const_scalar_node(name + '_0_s', np.int64(0), kwargs)
        create_const_scalar_node(name + '_1_s', np.int64(1), kwargs)
        nodes += [make_node('Cast', [length], [name + '_length'], to=int(TensorProto.INT64)), make_node('Cast', [name + '_0'], [name + '_0_itype'], to=dtype_t), make_node('Cast', [name + '_1'], [name + '_1_itype'], to=dtype_t), make_node('Div', [name + '_exp_out', name + '_rsum_out'], [name + '_div1_out']), make_node('Shape', [data], [name + '_shape0_out']), make_node('Shape', [name + '_shape0_out'], [name + '_in_dim']), make_node('Add', [name + '_in_dim', name + '_axis'], [name + '_dim+axis']), make_node('Less', [name + '_axis', name + '_0_s'], [name + '_less0_out']), make_node('Where', [name + '_less0_out', name + '_dim+axis', name + '_axis'], [name + '_final_axis']), make_node('Add', [name + '_final_axis', name + '_1_s'], [name + '_final_axis+1']), make_node('Slice', [name + '_shape0_out', name + '_final_axis', name + '_final_axis+1'], [name + '_axis_dim']), make_node('Squeeze', [name + '_axis_dim'], [name + '_axis_dim_s'], axes=[0]), make_node('Range', [name + '_0_s', name + '_axis_dim_s', name + '_1_s'], [name + '_range0_out']), make_node('Squeeze', [name + '_in_dim'], [name + '_in_dim_s'], axes=[0]), make_node('Range', [name + '_0_s', name + '_in_dim_s', name + '_1_s'], [name + '_range1_out']), make_node('Equal', [name + '_range1_out', name + '_final_axis'], [name + '_equal_out']), make_node('Cast', [name + '_equal_out'], [name + '_one_hot'], to=int(TensorProto.INT64)), make_node('Sub', [name + '_axis_dim_s', name + '_1_s'], [name + '_sub0_out']), make_node('Mul', [name + '_one_hot', name + '_sub0_out'], [name + '_mul0_out']), make_node('Add', [name + '_mul0_out', name + '_1_s'], [name + '_add0_out']), make_node('Reshape', [name + '_range0_out', name + '_add0_out'], [name + '_reshape0_out']), make_node('Mul', [name + '_one_hot', name + '_-1_s'], [name + '_mul1_out']), make_node('Add', [name + '_mul1_out', name + '_1_s'], [name + '_add1_out']), make_node('Sub', [name + '_shape0_out', name + '_1_s'], [name + '_sub1_out']), make_node('Mul', [name + '_add1_out', name + '_sub1_out'], [name + '_mul2_out']), make_node('Add', [name + '_mul2_out', name + '_1_s'], [name + '_add2_out']), make_node('Reshape', [name + '_length', name + '_add2_out'], [name + '_reshape1_out']), make_node('Less', [name + '_reshape0_out', name + '_reshape1_out'], [name + '_less_out']), make_node('Cast', [name + '_less_out'], [name + '_mask'], to=dtype_t), make_node('Mul', [name + '_div1_out', name + '_mask'], [name + '_mul3_out']), make_node('ReduceSum', [name + '_mul3_out'], [name + '_rsum1_out'], axes=[axis], keepdims=1), make_node('Equal', [name + '_rsum1_out', name + '_0_itype'], [name + '_equal1_out']), make_node('Where', [name + '_equal1_out', name + '_1_itype', name + '_rsum1_out'], [name + '_where_out']), make_node('Div', [name + '_mul3_out', name + '_where_out'], [name], name=name)]
        return nodes
    else:
        raise NotImplementedError('use_length must be true when both data and length are paased in.')