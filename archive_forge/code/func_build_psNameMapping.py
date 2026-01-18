from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging
def build_psNameMapping(self, ttFont):
    mapping = {}
    allNames = {}
    for i in range(ttFont['maxp'].numGlyphs):
        glyphName = psName = self.glyphOrder[i]
        if glyphName == '':
            glyphName = 'glyph%.5d' % i
        if glyphName in allNames:
            n = allNames[glyphName]
            while glyphName + '#' + str(n) in allNames:
                n += 1
            allNames[glyphName] = n + 1
            glyphName = glyphName + '#' + str(n)
        self.glyphOrder[i] = glyphName
        allNames[glyphName] = 1
        if glyphName != psName:
            mapping[glyphName] = psName
    self.mapping = mapping