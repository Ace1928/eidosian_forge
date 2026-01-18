import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_immediate_operand(self, code, value, dims):
    assert isinstance(dims, tuple)
    cache_key = (code, value)
    if cache_key not in self.cached_immediates:
        operand_id = len(self.operands)
        self.operands.append(Operand(code, dims, DimOrder.SCALAR_OR_VECTOR, 0.0, 0))
        self.values.append((operand_id, OperandValueSourceType.IMMEDIATE))
        self.value_data.append(value)
        self.cached_immediates[cache_key] = operand_id
    return self.cached_immediates[cache_key]