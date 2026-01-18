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
def getCoordinatesAndControls(self):
    coords = []
    controls = []
    if self.flags & VarComponentFlags.AXES_HAVE_VARIATION:
        for tag, v in self.location.items():
            controls.append(tag)
            coords.append((fl2fi(v, 14), 0))
    if self.flags & (VarComponentFlags.HAVE_TRANSLATE_X | VarComponentFlags.HAVE_TRANSLATE_Y):
        controls.append('translate')
        coords.append((self.transform.translateX, self.transform.translateY))
    if self.flags & VarComponentFlags.HAVE_ROTATION:
        controls.append('rotation')
        coords.append((fl2fi(self.transform.rotation / 180, 12), 0))
    if self.flags & (VarComponentFlags.HAVE_SCALE_X | VarComponentFlags.HAVE_SCALE_Y):
        controls.append('scale')
        coords.append((fl2fi(self.transform.scaleX, 10), fl2fi(self.transform.scaleY, 10)))
    if self.flags & (VarComponentFlags.HAVE_SKEW_X | VarComponentFlags.HAVE_SKEW_Y):
        controls.append('skew')
        coords.append((fl2fi(self.transform.skewX / -180, 12), fl2fi(self.transform.skewY / 180, 12)))
    if self.flags & (VarComponentFlags.HAVE_TCENTER_X | VarComponentFlags.HAVE_TCENTER_Y):
        controls.append('tCenter')
        coords.append((self.transform.tCenterX, self.transform.tCenterY))
    return (coords, controls)