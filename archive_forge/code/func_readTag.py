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
def readTag(self):
    pos = self.pos
    newpos = pos + 4
    value = Tag(self.data[pos:newpos])
    assert len(value) == 4, value
    self.pos = newpos
    return value