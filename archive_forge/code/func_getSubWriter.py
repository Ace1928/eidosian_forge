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
def getSubWriter(self):
    subwriter = self.__class__(self.localState, self.tableTag)
    subwriter.parent = self
    return subwriter