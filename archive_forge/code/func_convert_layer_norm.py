import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('LayerNorm')
def convert_layer_norm(node, **kwargs):
    """Map MXNet's LayerNorm operator attributes to onnx operators.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    axes = int(attrs.get('axis', -1))
    eps = attrs.get('eps', 9.99999975e-06)
    create_tensor([axes], name + '_axes', kwargs['initializer'])
    create_tensor([axes + 1], name + '_axes+1', kwargs['initializer'])
    create_const_scalar_node(name + '_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name + '_1_s', np.int64(1), kwargs)
    create_const_scalar_node(name + '_2_s', np.int64(2).astype(dtype), kwargs)
    create_const_scalar_node(name + '_eps', np.float32(eps), kwargs)
    nodes = [make_node('ReduceMean', [input_nodes[0]], [name + '_rm0_out'], axes=[axes]), make_node('Sub', [input_nodes[0], name + '_rm0_out'], [name + '_sub0_out']), make_node('Pow', [name + '_sub0_out', name + '_2_s'], [name + '_pow0_out']), make_node('ReduceMean', [name + '_pow0_out'], [name + '_rm1_out'], axes=[axes]), make_node('Add', [name + '_rm1_out', name + '_eps'], [name + '_add0_out']), make_node('Sqrt', [name + '_add0_out'], [name + '_sqrt0_out']), make_node('Div', [name + '_sub0_out', name + '_sqrt0_out'], [name + '_div0_out'])]
    if axes == -1:
        nodes += [make_node('Mul', [name + '_div0_out', input_nodes[1]], [name + '_mul0_out']), make_node('Neg', [input_nodes[2]], [name + '_neg']), make_node('Sub', [name + '_mul0_out', name + '_neg'], [name])]
    else:
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_shape0_out']), make_node('Shape', [name + '_shape0_out'], [name + '_in_dim']), make_node('Squeeze', [name + '_in_dim'], [name + '_in_dim_s'], axes=[0]), make_node('Range', [name + '_0_s', name + '_in_dim_s', name + '_1_s'], [name + '_range']), make_node('Equal', [name + '_range', name + '_axes'], [name + '_equal']), make_node('Cast', [name + '_equal'], [name + '_one_hot'], to=int(TensorProto.INT64)), make_node('Slice', [name + '_shape0_out', name + '_axes', name + '_axes+1'], [name + '_slice_out']), make_node('Squeeze', [name + '_slice_out'], [name + '_slice_out_s'], axes=[0]), make_node('Sub', [name + '_slice_out_s', name + '_1_s'], [name + '_sub1_out']), make_node('Mul', [name + '_one_hot', name + '_sub1_out'], [name + '_mul0_out']), make_node('Add', [name + '_mul0_out', name + '_1_s'], [name + '_add1_out']), make_node('Reshape', [input_nodes[1], name + '_add1_out'], [name + 'gamma_exp']), make_node('Reshape', [input_nodes[2], name + '_add1_out'], [name + 'beta_exp']), make_node('Expand', [name + 'gamma_exp', name + '_shape0_out'], [name + 'gamma_exp1']), make_node('Expand', [name + 'beta_exp', name + '_shape0_out'], [name + 'beta_exp1']), make_node('Mul', [name + '_div0_out', name + 'gamma_exp1'], [name + '_mul1_out']), make_node('Add', [name + '_mul1_out', name + 'beta_exp1'], [name], name=name)]
    return nodes