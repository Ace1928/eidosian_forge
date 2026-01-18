import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_pointwise_simple_unary_op(self, node, opcode):
    assert node.inputsSize() == 1
    assert node.outputsSize() == 1
    in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
    out_oper = in_oper
    if opcode == NNAPI_OperationCode.LOGISTIC:
        if in_oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
            out_oper = in_oper._replace(zero_point=0, scale=1.0 / 256)
    out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)
    for idx, dim in enumerate(in_oper.shape):
        if dim == 0:
            self.forward_operand_shape(out_id, idx, in_id, idx)
    inputs = [None] * 1
    inputs[0] = in_id
    outputs = [None] * 1
    outputs[0] = out_id
    self.add_operation(opcode, inputs, outputs)