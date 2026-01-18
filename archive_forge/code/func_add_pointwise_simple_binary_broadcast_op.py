import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_pointwise_simple_binary_broadcast_op(self, node, opcode, fuse_code):
    assert node.inputsSize() == 2
    self._do_add_binary(node, opcode, fuse_code)