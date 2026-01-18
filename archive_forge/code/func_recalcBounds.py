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
def recalcBounds(self, glyfTable, *, boundsDone=None):
    """Recalculates the bounds of the glyph.

        Each glyph object stores its bounding box in the
        ``xMin``/``yMin``/``xMax``/``yMax`` attributes. These bounds must be
        recomputed when the ``coordinates`` change. The ``table__g_l_y_f`` bounds
        must be provided to resolve component bounds.
        """
    if self.isComposite() and self.tryRecalcBoundsComposite(glyfTable, boundsDone=boundsDone):
        return
    try:
        coords, endPts, flags = self.getCoordinates(glyfTable)
        self.xMin, self.yMin, self.xMax, self.yMax = coords.calcIntBounds()
    except NotImplementedError:
        pass