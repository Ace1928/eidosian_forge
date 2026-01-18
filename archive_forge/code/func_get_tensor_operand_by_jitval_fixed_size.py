import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_tensor_operand_by_jitval_fixed_size(self, jitval):
    op_id, oper = self.get_tensor_operand_by_jitval(jitval)
    for s in oper.shape:
        if s == 0:
            raise Exception('Flexible size is not supported for this operand.')
        if s < 0:
            LOG.warning('Operand %s has runtime flex shape', oper)
    return (op_id, oper)