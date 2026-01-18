from collections import namedtuple
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set
def getComponentNames(self, glyfTable):
    """Returns a list of names of component glyphs used in this glyph

        This method can be used on simple glyphs (in which case it returns an
        empty list) or composite glyphs.
        """
    if hasattr(self, 'data') and self.isVarComposite():
        self.expand(glyfTable)
    if not hasattr(self, 'data'):
        if self.isComposite() or self.isVarComposite():
            return [c.glyphName for c in self.components]
        else:
            return []
    if not self.data or struct.unpack('>h', self.data[:2])[0] >= 0:
        return []
    data = self.data
    i = 10
    components = []
    more = 1
    while more:
        flags, glyphID = struct.unpack('>HH', data[i:i + 4])
        i += 4
        flags = int(flags)
        components.append(glyfTable.getGlyphName(int(glyphID)))
        if flags & ARG_1_AND_2_ARE_WORDS:
            i += 4
        else:
            i += 2
        if flags & WE_HAVE_A_SCALE:
            i += 2
        elif flags & WE_HAVE_AN_X_AND_Y_SCALE:
            i += 4
        elif flags & WE_HAVE_A_TWO_BY_TWO:
            i += 8
        more = flags & MORE_COMPONENTS
    return components