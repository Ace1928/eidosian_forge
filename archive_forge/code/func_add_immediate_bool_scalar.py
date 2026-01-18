import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_immediate_bool_scalar(self, value):
    return self.add_immediate_operand(NNAPI_OperandCode.BOOL, b'\x01' if value else b'\x00', ())