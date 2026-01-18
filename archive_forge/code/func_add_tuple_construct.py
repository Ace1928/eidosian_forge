import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_tuple_construct(self, node):
    assert node.outputsSize() == 1
    output = node.outputsAt(0)
    values = []
    for inp in node.inputs():
        values.append(inp)
    self.add_tensor_sequence(output, values)