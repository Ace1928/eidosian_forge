import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def compute_operand_shape(self, op_id, dim, expr):
    self.flexible_shape_computation_lines.append(f'{flex_name(op_id, dim)} = {expr}')