import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_pool2d_node(self, node, opcode):
    assert node.inputsSize() == 6
    assert node.outputsSize() == 1
    image, kernel, stride, padding, dilation, ceil_mode = node.inputs()
    stride = stride or kernel
    args = self.get_conv_pool_args_2d_from_jit(self.get_size_arg(kernel), stride, padding, dilation)
    if args.dilation_h != 1 or args.dilation_w != 1:
        raise Exception('NNAPI does not support dilated pooling.')
    image_id, image_oper = self.get_tensor_operand_by_jitval_fixed_size(image)
    assert len(image_oper.shape) == 4
    out_shape = get_conv_pool_shape(image_oper.shape, args, image_oper.shape[1], False)
    use_nchw = image_oper.use_nchw()
    inputs = [None] * 11
    inputs[0] = image_id
    inputs[1] = self.add_immediate_int_scalar(args.pad_l)
    inputs[2] = self.add_immediate_int_scalar(args.pad_r)
    inputs[3] = self.add_immediate_int_scalar(args.pad_t)
    inputs[4] = self.add_immediate_int_scalar(args.pad_b)
    inputs[5] = self.add_immediate_int_scalar(args.stride_w)
    inputs[6] = self.add_immediate_int_scalar(args.stride_h)
    inputs[7] = self.add_immediate_int_scalar(args.kernel_w)
    inputs[8] = self.add_immediate_int_scalar(args.kernel_h)
    inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
    inputs[10] = self.add_immediate_bool_scalar(use_nchw)
    outputs = [None] * 1
    outputs[0] = self.add_tensor_operand(node.outputsAt(0), image_oper._replace(shape=out_shape))
    self.add_operation(opcode, inputs, outputs)