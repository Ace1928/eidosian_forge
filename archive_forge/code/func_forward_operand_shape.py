import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def forward_operand_shape(self, out_op_id, out_dim, in_op_id, in_dim):
    self.compute_operand_shape(out_op_id, out_dim, flex_name(in_op_id, in_dim))