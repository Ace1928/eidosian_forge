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
def getDataForHarfbuzz(self):
    """Assemble the data for this writer/table with all offset field set to 0"""
    items = list(self.items)
    packFuncs = {2: packUShort, 3: packUInt24, 4: packULong}
    for i, item in enumerate(items):
        if hasattr(item, 'subWriter'):
            if item.offsetSize in packFuncs:
                items[i] = packFuncs[item.offsetSize](0)
            else:
                raise ValueError(item.offsetSize)
    return bytesjoin(items)