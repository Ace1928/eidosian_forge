import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Pad')
def convert_pad(node, **kwargs):
    """Map MXNet's pad operator attributes to onnx's Pad operator
    and return the created node.
    """
    from onnx.helper import make_node
    opset_version = kwargs['opset_version']
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    mxnet_pad_width = convert_string_to_list(attrs.get('pad_width'))
    onnx_pad_width = transform_padding(mxnet_pad_width)
    pad_mode = attrs.get('mode')
    pad_value = float(attrs.get('constant_value', 0.0))
    pad_value = dtype.type(pad_value)
    if opset_version >= 11:
        create_const_node(name + '_pads', np.array(onnx_pad_width, dtype='int64'), kwargs)
        nodes = []
        if pad_mode == 'constant':
            create_const_scalar_node(name + '_const', pad_value, kwargs)
            nodes += [make_node('Pad', [input_nodes[0], name + '_pads', name + '_const'], [name], mode=pad_mode, name=name)]
        else:
            nodes += [make_node('Pad', [input_nodes[0], name + '_pads'], [name], mode=pad_mode, name=name)]
        return nodes
    else:
        if pad_mode == 'constant':
            node = onnx.helper.make_node('Pad', inputs=input_nodes, outputs=[name], mode='constant', value=pad_value, pads=onnx_pad_width, name=name)
        else:
            node = onnx.helper.make_node('Pad', inputs=input_nodes, outputs=[name], mode=pad_mode, pads=onnx_pad_width, name=name)
        return [node]