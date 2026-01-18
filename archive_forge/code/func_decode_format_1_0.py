from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging
def decode_format_1_0(self, data, ttFont):
    self.glyphOrder = standardGlyphOrder[:ttFont['maxp'].numGlyphs]