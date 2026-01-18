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
def getCompositeMaxpValues(self, glyfTable, maxComponentDepth=1):
    assert self.isComposite() or self.isVarComposite()
    nContours = 0
    nPoints = 0
    initialMaxComponentDepth = maxComponentDepth
    for compo in self.components:
        baseGlyph = glyfTable[compo.glyphName]
        if baseGlyph.numberOfContours == 0:
            continue
        elif baseGlyph.numberOfContours > 0:
            nP, nC = baseGlyph.getMaxpValues()
        else:
            nP, nC, componentDepth = baseGlyph.getCompositeMaxpValues(glyfTable, initialMaxComponentDepth + 1)
            maxComponentDepth = max(maxComponentDepth, componentDepth)
        nPoints = nPoints + nP
        nContours = nContours + nC
    return CompositeMaxpValues(nPoints, nContours, maxComponentDepth)