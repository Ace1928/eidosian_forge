import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
@staticmethod
def serialize_ints(ints):
    return array.array('i', ints).tobytes()