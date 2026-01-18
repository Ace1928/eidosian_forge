import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_prelu_op(self, node):
    assert node.inputsSize() == 2
    assert node.outputsSize() == 1
    assert node.inputsAt(0).type().kind() == 'TensorType'
    assert node.inputsAt(1).type().kind() == 'TensorType'
    in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
    w_id, w_oper = self.get_tensor_operand_for_weight(node.inputsAt(1))
    assert len(w_oper.shape) == 1
    assert w_oper.shape[0] > 0
    if w_oper.shape[0] > 1:
        if in_oper.use_nchw():
            raise Exception('Per-channel PReLU only supports channels_last right now.')
    out_id = self.add_tensor_operand(node.outputsAt(0), in_oper)
    for dim, size in enumerate(in_oper.shape):
        if size > 0:
            pass
        elif dim <= 1:
            raise Exception('PReLU requires fixed size for dim 0 and dim 1.')
        else:
            self.forward_operand_shape(out_id, dim, in_id, dim)
    inputs = [None] * 2
    inputs[0] = in_id
    inputs[1] = w_id
    outputs = [None] * 1
    outputs[0] = out_id
    self.add_operation(NNAPI_OperationCode.PRELU, inputs, outputs)