import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def has_operand_for_jitval(self, jitval):
    return jitval in self.jitval_operand_map