import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_contrib_ROIAlign')
def convert_contrib_roialign(node, **kwargs):
    """Map MXNet's _contrib_ROIAlign
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    pooled_size = convert_string_to_list(str(attrs.get('pooled_size')))
    spatial_scale = float(attrs.get('spatial_scale'))
    sample_ratio = int(attrs.get('sample_ratio', '0'))
    position_sensitive = attrs.get('position_sensitive', 'False')
    aligned = attrs.get('aligned', 'False')
    if position_sensitive != 'False':
        raise NotImplementedError('_contrib_ROIAlign does not currently support                                    position_sensitive!=False')
    if aligned != 'False':
        raise NotImplementedError('_contrib_ROIAlign does not currently support                                    aligned!=False')
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([0], name + '_0_s', kwargs['initializer'], dtype='float32')
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([5], name + '_5', kwargs['initializer'])
    nodes = [make_node('Slice', [input_nodes[1], name + '_1', name + '_5', name + '_1'], [name + '_rois']), make_node('Slice', [input_nodes[1], name + '_0', name + '_1', name + '_1'], [name + '_inds___']), make_node('Squeeze', [name + '_inds___'], [name + '_inds__'], axes=[1]), make_node('Relu', [name + '_inds__'], [name + '_inds_']), make_node('Cast', [name + '_inds_'], [name + '_inds'], to=int(TensorProto.INT64)), make_node('RoiAlign', [input_nodes[0], name + '_rois', name + '_inds'], [name + '_roi'], mode='avg', output_height=pooled_size[0], output_width=pooled_size[1], sampling_ratio=sample_ratio, spatial_scale=spatial_scale), make_node('Unsqueeze', [name + '_inds___'], [name + '_unsq'], axes=(2, 3)), make_node('Less', [name + '_unsq', name + '_0_s'], [name + '_less']), make_node('Where', [name + '_less', name + '_0_s', name + '_roi'], [name])]
    return nodes