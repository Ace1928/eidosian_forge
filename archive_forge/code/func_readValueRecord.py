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
def readValueRecord(self, reader, font):
    format = self.format
    if not format:
        return None
    valueRecord = ValueRecord()
    for name, isDevice, signed in format:
        if signed:
            value = reader.readShort()
        else:
            value = reader.readUShort()
        if isDevice:
            if value:
                from . import otTables
                subReader = reader.getSubReader(value)
                value = getattr(otTables, name)()
                value.decompile(subReader, font)
            else:
                value = None
        setattr(valueRecord, name, value)
    return valueRecord