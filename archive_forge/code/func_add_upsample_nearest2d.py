import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_upsample_nearest2d(self, node):
    assert node.inputsSize() == 3 or node.inputsSize() == 4
    assert node.outputsSize() == 1
    if node.inputsSize() == 3:
        image, size_jit, scale_jit = node.inputs()
    else:
        image, size_jit, scale_h_jit, scale_w_jit = node.inputs()
    size_ctype, size_arg = self.get_constant_value(size_jit)
    if node.inputsSize() == 3:
        scale_ctype, scale_arg = self.get_constant_value(scale_jit)
    else:
        scale_h_ctype, scale_h_arg = self.get_constant_value(scale_h_jit)
        scale_w_ctype, scale_w_arg = self.get_constant_value(scale_w_jit)
        assert scale_h_ctype.kind() == 'NoneType'
        assert scale_w_ctype.kind() == 'NoneType'
        scale_ctype = scale_h_ctype
        scale_arg = scale_h_arg
    image_id, image_oper = self.get_tensor_operand_by_jitval(image)
    assert len(image_oper.shape) == 4
    if size_ctype.kind() != 'NoneType' and scale_ctype.kind() != 'NoneType':
        raise Exception('Size and scale cannot both be non-None.')
    elif size_ctype.kind() != 'NoneType':
        assert size_ctype.kind() == 'ListType'
        assert size_ctype.getElementType().kind() == 'IntType'
        assert scale_ctype.kind() == 'NoneType'
        assert scale_arg is None
        assert isinstance(size_arg, list)
        assert size_arg
        assert all((isinstance(val, int) for val in size_arg))
        if len(size_arg) == 1:
            size_arg = size_arg * 2
        assert len(size_arg) == 2
        out_h = size_arg[0]
        out_w = size_arg[1]
        arg_h = self.add_immediate_int_scalar(out_h)
        arg_w = self.add_immediate_int_scalar(out_w)
    elif scale_ctype.kind() != 'NoneType':
        assert scale_ctype.kind() == 'ListType'
        assert scale_ctype.getElementType().kind() == 'FloatType'
        assert size_ctype.kind() == 'NoneType'
        assert size_arg is None
        assert isinstance(scale_arg, list)
        assert scale_arg
        assert all((isinstance(val, float) for val in scale_arg))
        if len(scale_arg) == 1:
            scale_arg = scale_arg * 2
        assert len(scale_arg) == 2
        out_h = int(scale_arg[0] * image_oper.shape[2])
        out_w = int(scale_arg[1] * image_oper.shape[3])
        arg_h = self.add_immediate_float_scalar(scale_arg[0])
        arg_w = self.add_immediate_float_scalar(scale_arg[1])
    else:
        raise Exception('Size and scale cannot both be None.')
    out_shape = (image_oper.shape[0], image_oper.shape[1], out_h, out_w)
    use_nchw = image_oper.use_nchw()
    out_id = self.add_tensor_operand(node.outputsAt(0), image_oper._replace(shape=out_shape))
    if image_oper.shape[0] == 0 or image_oper.shape[1] == 0:
        raise Exception('Flexible batch or channels not supported')
    for dim in (2, 3):
        if image_oper.shape[dim] == 0:
            if size_ctype.kind() != 'NoneType':
                self.compute_operand_shape(out_id, dim, size_arg[dim - 2])
            elif scale_ctype.kind() != 'NoneType':
                self.compute_operand_shape(out_id, dim, f'int({scale_arg[dim - 2]} * {flex_name(image_id, dim)})')
            else:
                raise Exception('Size and scale cannot both be None.')
    inputs = [None] * 4
    inputs[0] = image_id
    inputs[1] = arg_w
    inputs[2] = arg_h
    inputs[3] = self.add_immediate_bool_scalar(use_nchw)
    outputs = [None] * 1
    outputs[0] = out_id
    self.add_operation(NNAPI_OperationCode.RESIZE_NEAREST_NEIGHBOR, inputs, outputs)