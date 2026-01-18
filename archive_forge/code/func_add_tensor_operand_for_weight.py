import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_tensor_operand_for_weight(self, tensor, dim_order=DimOrder.UNKNOWN_CONSTANT):
    toper = self.torch_tensor_to_operand(tensor, dim_order)
    operand_id = len(self.operands)
    self.operands.append(toper)
    tsize = tensor_size(toper.op_type, toper.shape)
    psize = (tsize - 1 | 3) + 1
    self.values.append((operand_id, OperandValueSourceType.NUMBERED_BUFFER))
    buf_num = len(self.used_weights)
    offset = 0
    self.value_data.append(struct.pack('iii', buf_num, offset, tsize))
    if dim_order == DimOrder.CHANNELS_LAST:
        tensor = tensor.permute(0, 2, 3, 1)
    self.used_weights.append(tensor)
    return operand_id