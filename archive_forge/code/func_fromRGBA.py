from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import array
from collections import namedtuple
import struct
import sys
@classmethod
def fromRGBA(cls, red, green, blue, alpha):
    return cls(red=red, green=green, blue=blue, alpha=alpha)