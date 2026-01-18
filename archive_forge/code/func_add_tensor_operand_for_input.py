import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_tensor_operand_for_input(self, arg_idx, jitval, tensor):
    dim_order = DimOrder.CHANNELS_LAST if getattr(tensor, 'nnapi_nhwc', False) else DimOrder.PRESUMED_CONTIGUOUS
    toper = self.torch_tensor_to_operand(tensor, dim_order)
    operand_id = self.add_tensor_operand(jitval, toper)
    self.inputs.append(operand_id)
    for dim, size in enumerate(tensor.shape):
        if size == 0:
            self.compute_operand_shape(operand_id, dim, f'args[{arg_idx}].shape[{dim}]')
    return operand_id