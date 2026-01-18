import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_tensor_operand_for_weight(self, jitval):
    _, value = self.get_constant_value(jitval, 'TensorType')
    operand_id = self.add_tensor_operand_for_weight(value)
    return (operand_id, self.operands[operand_id])