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
def decompileVarGlyph(glyphName, gid):
    gvarData = data[offsetToData + offsets[gid]:offsetToData + offsets[gid + 1]]
    if not gvarData:
        return []
    glyph = glyf[glyphName]
    numPointsInGlyph = self.getNumPoints_(glyph)
    return decompileGlyph_(numPointsInGlyph, sharedCoords, axisTags, gvarData)