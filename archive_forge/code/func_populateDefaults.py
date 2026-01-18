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
def populateDefaults(self, propagator=None):
    for conv in self.getConverters():
        if conv.repeat:
            if not hasattr(self, conv.name):
                setattr(self, conv.name, [])
            countValue = len(getattr(self, conv.name)) - conv.aux
            try:
                count_conv = self.getConverterByName(conv.repeat)
                setattr(self, conv.repeat, countValue)
            except KeyError:
                if propagator and conv.repeat in propagator:
                    propagator[conv.repeat].setValue(countValue)
        else:
            if conv.aux and (not eval(conv.aux, None, self.__dict__)):
                continue
            if hasattr(self, conv.name):
                continue
            if hasattr(conv, 'writeNullOffset'):
                setattr(self, conv.name, None)
            if hasattr(conv, 'DEFAULT'):
                setattr(self, conv.name, conv.DEFAULT)