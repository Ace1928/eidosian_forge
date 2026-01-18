from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging
def decode_format_4_0(self, data, ttFont):
    from fontTools import agl
    numGlyphs = ttFont['maxp'].numGlyphs
    indices = array.array('H')
    indices.frombytes(data)
    if sys.byteorder != 'big':
        indices.byteswap()
    self.glyphOrder = glyphOrder = [''] * int(numGlyphs)
    for i in range(min(len(indices), numGlyphs)):
        if indices[i] == 65535:
            self.glyphOrder[i] = ''
        elif indices[i] in agl.UV2AGL:
            self.glyphOrder[i] = agl.UV2AGL[indices[i]]
        else:
            self.glyphOrder[i] = 'uni%04X' % indices[i]
    self.build_psNameMapping(ttFont)