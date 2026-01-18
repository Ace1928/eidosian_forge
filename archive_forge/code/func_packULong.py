from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
def packULong(value):
    assert 0 <= value < 4294967296, value
    return struct.pack('>I', value)