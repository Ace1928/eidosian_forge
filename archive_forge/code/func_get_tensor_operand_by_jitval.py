import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_tensor_operand_by_jitval(self, jitval):
    operand_id = self.jitval_operand_map[jitval]
    return (operand_id, self.operands[operand_id])