import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_unsqueeze(self, node):
    assert node.inputsSize() == 2
    assert node.outputsSize() == 1
    in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
    _, dim = self.get_constant_value(node.inputsAt(1), 'IntType')
    assert in_oper.dim_order == DimOrder.PRESUMED_CONTIGUOUS
    real_dim = dim if dim >= 0 else dim + len(in_oper.shape) + 1
    out_shape_list = list(in_oper.shape)
    out_shape_list.insert(real_dim, 1)
    out_shape = tuple(out_shape_list)
    out_oper = in_oper._replace(shape=out_shape)
    inputs = [None] * 2
    inputs[0] = in_id
    inputs[1] = self.add_immediate_int_scalar(dim)
    outputs = [None] * 1
    outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)
    self.add_operation(NNAPI_OperationCode.EXPAND_DIMS, inputs, outputs)