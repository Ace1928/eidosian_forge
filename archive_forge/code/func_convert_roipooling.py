import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('ROIPooling')
def convert_roipooling(node, **kwargs):
    """Map MXNet's ROIPooling operator attributes to onnx's MaxRoiPool
    operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    pooled_shape = convert_string_to_list(attrs.get('pooled_size'))
    scale = float(attrs.get('spatial_scale'))
    node = onnx.helper.make_node('MaxRoiPool', input_nodes, [name], pooled_shape=pooled_shape, spatial_scale=scale, name=name)
    return [node]