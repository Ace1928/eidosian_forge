import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_slice(self, node):
    assert node.inputsSize() == 5
    assert node.outputsSize() == 1
    in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
    _, dim_value = self.get_constant_value(node.inputsAt(1))
    _, start_value = self.get_constant_value(node.inputsAt(2))
    _, stop_value = self.get_constant_value(node.inputsAt(3))
    _, step_value = self.get_constant_value(node.inputsAt(4))
    if start_value is None:
        start_value = 0
    if stop_value is None:
        stop_value = sys.maxsize
    if start_value < 0:
        start_value += in_oper.shape[dim_value]
    elif start_value == sys.maxsize:
        start_value = 0
    if start_value == 0 and stop_value == sys.maxsize:
        self._identity(node)
        return
    if in_oper.shape[dim_value] == 0:
        raise Exception('Unable to slice with flexible shape')
    if stop_value < 0:
        stop_value += in_oper.shape[dim_value]
    elif stop_value == sys.maxsize:
        stop_value = in_oper.shape[dim_value]
    if start_value >= stop_value:
        raise Exception('Slice start value should be less than stop value')
    out_len = (stop_value - start_value) // step_value
    out_shape = tuple((out_len if i == dim_value else dim for i, dim in enumerate(in_oper.shape)))
    out_id = self.add_tensor_operand(node.outputsAt(0), in_oper._replace(shape=out_shape))
    end_mask = 0
    for idx, dim in enumerate(out_shape):
        if dim == 0:
            self.forward_operand_shape(out_id, idx, in_id, idx)
            end_mask |= 1 << idx
    inputs = [None] * 7
    inputs[0] = in_id
    inputs[1] = self.add_immediate_int_vector([start_value if i == dim_value else 0 for i in range(len(in_oper.shape))])
    inputs[2] = self.add_immediate_int_vector([stop_value if i == dim_value else dim for i, dim in enumerate(in_oper.shape)])
    inputs[3] = self.add_immediate_int_vector([step_value if i == dim_value else 1 for i in range(len(in_oper.shape))])
    inputs[4] = self.add_immediate_int_scalar(0)
    inputs[5] = self.add_immediate_int_scalar(end_mask)
    inputs[6] = self.add_immediate_int_scalar(0)
    outputs = [None] * 1
    outputs[0] = out_id
    self.add_operation(NNAPI_OperationCode.STRIDED_SLICE, inputs, outputs)