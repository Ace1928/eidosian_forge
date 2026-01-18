from collections import UserDict, deque
from functools import partial
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
import array
import itertools
import logging
import struct
import sys
import fontTools.ttLib.tables.TupleVariation as tv
@staticmethod
def decompileOffsets_(data, tableFormat, glyphCount):
    if tableFormat == 0:
        offsets = array.array('H')
        offsetsSize = (glyphCount + 1) * 2
    else:
        offsets = array.array('I')
        offsetsSize = (glyphCount + 1) * 4
    offsets.frombytes(data[0:offsetsSize])
    if sys.byteorder != 'big':
        offsets.byteswap()
    if tableFormat == 0:
        offsets = [off * 2 for off in offsets]
    return offsets