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
def setCoordinates(self, coords):
    i = 0
    if self.flags & VarComponentFlags.AXES_HAVE_VARIATION:
        newLocation = {}
        for tag in self.location:
            newLocation[tag] = fi2fl(coords[i][0], 14)
            i += 1
        self.location = newLocation
    self.transform = DecomposedTransform()
    if self.flags & (VarComponentFlags.HAVE_TRANSLATE_X | VarComponentFlags.HAVE_TRANSLATE_Y):
        self.transform.translateX, self.transform.translateY = coords[i]
        i += 1
    if self.flags & VarComponentFlags.HAVE_ROTATION:
        self.transform.rotation = fi2fl(coords[i][0], 12) * 180
        i += 1
    if self.flags & (VarComponentFlags.HAVE_SCALE_X | VarComponentFlags.HAVE_SCALE_Y):
        self.transform.scaleX, self.transform.scaleY = (fi2fl(coords[i][0], 10), fi2fl(coords[i][1], 10))
        i += 1
    if self.flags & (VarComponentFlags.HAVE_SKEW_X | VarComponentFlags.HAVE_SKEW_Y):
        self.transform.skewX, self.transform.skewY = (fi2fl(coords[i][0], 12) * -180, fi2fl(coords[i][1], 12) * 180)
        i += 1
    if self.flags & (VarComponentFlags.HAVE_TCENTER_X | VarComponentFlags.HAVE_TCENTER_Y):
        self.transform.tCenterX, self.transform.tCenterY = coords[i]
        i += 1
    return coords[i:]