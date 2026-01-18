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
def compileCoordinates(self):
    assert len(self.coordinates) == len(self.flags)
    data = []
    endPtsOfContours = array.array('H', self.endPtsOfContours)
    if sys.byteorder != 'big':
        endPtsOfContours.byteswap()
    data.append(endPtsOfContours.tobytes())
    instructions = self.program.getBytecode()
    data.append(struct.pack('>h', len(instructions)))
    data.append(instructions)
    deltas = self.coordinates.copy()
    deltas.toInt()
    deltas.absoluteToRelative()
    deltas = self.compileDeltasGreedy(self.flags, deltas)
    data.extend(deltas)
    return b''.join(data)