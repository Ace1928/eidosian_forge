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
@deprecateFunction("use '_getPhantomPoints' instead", category=DeprecationWarning)
def getPhantomPoints(self, glyphName, ttFont, defaultVerticalOrigin=None):
    """Old public name for self._getPhantomPoints().
        See: https://github.com/fonttools/fonttools/pull/2266"""
    hMetrics = ttFont['hmtx'].metrics
    vMetrics = self._synthesizeVMetrics(glyphName, ttFont, defaultVerticalOrigin)
    return self._getPhantomPoints(glyphName, hMetrics, vMetrics)