import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_tensor_operand(self, jitval, oper):
    assert isinstance(oper, Operand)
    if jitval in self.jitval_operand_map:
        raise Exception(f'Duplicate tensor: {jitval!r}')
    operand_id = self.get_next_operand_id()
    self.operands.append(oper)
    self.jitval_operand_map[jitval] = operand_id
    return operand_id