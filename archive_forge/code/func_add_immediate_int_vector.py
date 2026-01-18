import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_immediate_int_vector(self, value):
    return self.add_immediate_operand(NNAPI_OperandCode.TENSOR_INT32, array.array('i', value).tobytes(), (len(value),))