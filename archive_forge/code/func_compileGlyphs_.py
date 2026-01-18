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
def compileGlyphs_(self, ttFont, axisTags, sharedCoordIndices):
    result = []
    glyf = ttFont['glyf']
    for glyphName in ttFont.getGlyphOrder():
        variations = self.variations.get(glyphName, [])
        if not variations:
            result.append(b'')
            continue
        pointCountUnused = 0
        result.append(compileGlyph_(variations, pointCountUnused, axisTags, sharedCoordIndices))
    return result