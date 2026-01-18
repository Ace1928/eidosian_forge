import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_log_softmax(self, node):
    assert node.inputsSize() == 3
    assert node.outputsSize() == 1
    jit_input, jit_dim, jit_half_to_float = node.inputs()
    input_id, input_oper = self.get_tensor_operand_by_jitval_fixed_size(jit_input)
    _, dim = self.get_constant_value(jit_dim, 'IntType')
    out_shape = input_oper.shape
    inputs = [None] * 3
    inputs[0] = input_id
    inputs[1] = self.add_immediate_float_scalar(1)
    inputs[2] = self.add_immediate_int_scalar(dim)
    outputs = [None] * 1
    outputs[0] = self.add_tensor_operand(node.outputsAt(0), input_oper._replace(shape=out_shape))
    self.add_operation(NNAPI_OperationCode.LOG_SOFTMAX, inputs, outputs)