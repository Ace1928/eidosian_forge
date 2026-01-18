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
def flagBest(x, y, onCurve):
    """For a given x,y delta pair, returns the flag that packs this pair
    most efficiently, as well as the number of byte cost of such flag."""
    flag = flagOnCurve if onCurve else 0
    cost = 0
    if x == 0:
        flag = flag | flagXsame
    elif -255 <= x <= 255:
        flag = flag | flagXShort
        if x > 0:
            flag = flag | flagXsame
        cost += 1
    else:
        cost += 2
    if y == 0:
        flag = flag | flagYsame
    elif -255 <= y <= 255:
        flag = flag | flagYShort
        if y > 0:
            flag = flag | flagYsame
        cost += 1
    else:
        cost += 2
    return (flag, cost)