from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging
def encode_format_2_0(self, ttFont):
    numGlyphs = ttFont['maxp'].numGlyphs
    glyphOrder = ttFont.getGlyphOrder()
    assert len(glyphOrder) == numGlyphs
    indices = array.array('H')
    extraDict = {}
    extraNames = self.extraNames = [n for n in self.extraNames if n not in standardGlyphOrder]
    for i in range(len(extraNames)):
        extraDict[extraNames[i]] = i
    for glyphID in range(numGlyphs):
        glyphName = glyphOrder[glyphID]
        if glyphName in self.mapping:
            psName = self.mapping[glyphName]
        else:
            psName = glyphName
        if psName in extraDict:
            index = 258 + extraDict[psName]
        elif psName in standardGlyphOrder:
            index = standardGlyphOrder.index(psName)
        else:
            index = 258 + len(extraNames)
            extraDict[psName] = len(extraNames)
            extraNames.append(psName)
        indices.append(index)
    if sys.byteorder != 'big':
        indices.byteswap()
    return struct.pack('>H', numGlyphs) + indices.tobytes() + packPStrings(extraNames)