import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_optional_bias(self, jit_bias, weight_tensor, transpose=False):
    ctype, value = self.get_constant_value(jit_bias)
    if ctype.kind() == 'NoneType':
        bias_idx = 1 if transpose else 0
        nnapi_bias_tensor = torch.zeros(weight_tensor.size()[bias_idx], dtype=weight_tensor.dtype)
        bias_id = self.add_tensor_operand_for_weight(nnapi_bias_tensor)
        bias_oper = self.operands[bias_id]
        return (bias_id, bias_oper)
    else:
        return self.get_tensor_operand_for_weight(jit_bias)