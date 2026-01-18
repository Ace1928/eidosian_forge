import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_contrib_BilinearResize2D')
def convert_contrib_BilinearResize2D(node, **kwargs):
    """Map MXNet's contrib_BilinearResize2D operator attributes to onnx.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')
    height = int(attrs.get('height', 0))
    width = int(attrs.get('width', 0))
    scale_height = float(attrs.get('scale_height', 0))
    scale_width = float(attrs.get('scale_width', 0))
    if height * width == 0 and scale_height * scale_width == 0:
        raise AttributeError('height, width or scale_height, scale_width cannot be 0')
    mode = attrs.get('mode', 'size')
    if mode != 'size':
        raise NotImplementedError('contrib_BilinearResize2D with mode other than "size" is                                    not supported')
    create_tensor([], name + '_roi', kwargs['initializer'], dtype='float32')
    create_tensor([], name + '_scales_empty', kwargs['initializer'], dtype='float32')
    nodes = []
    if scale_height == 0:
        create_tensor([0], name + '_0', kwargs['initializer'])
        create_tensor([2], name + '_2', kwargs['initializer'])
        create_tensor([height, width], name + '_h_w', kwargs['initializer'], dtype='int64')
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Slice', [name + '_shape', name + '_0', name + '_2'], [name + '_shape_01']), make_node('Concat', [name + '_shape_01', name + '_h_w'], [name + '_sizes'], axis=0)]
    else:
        create_tensor([1, 1, scale_height, scale_width], name + '_scales', kwargs['initializer'], dtype='float32')
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Cast', [name + '_shape'], [name + '_shape_f'], to=int(TensorProto.FLOAT)), make_node('Mul', [name + '_shape_f', name + '_scales'], [name + '_sizes_']), make_node('Cast', [name + '_sizes_'], [name + '_sizes'], to=int(TensorProto.INT64))]
    nodes += [make_node('Resize', [input_nodes[0], name + '_roi', name + '_scales_empty', name + '_sizes'], [name], mode='linear', coordinate_transformation_mode='align_corners', name=name)]
    return nodes