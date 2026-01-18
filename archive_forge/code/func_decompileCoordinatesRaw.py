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
def decompileCoordinatesRaw(self, nCoordinates, data, pos=0):
    flags = bytearray(nCoordinates)
    xFormat = '>'
    yFormat = '>'
    j = 0
    while True:
        flag = data[pos]
        pos += 1
        repeat = 1
        if flag & flagRepeat:
            repeat = data[pos] + 1
            pos += 1
        for k in range(repeat):
            if flag & flagXShort:
                xFormat = xFormat + 'B'
            elif not flag & flagXsame:
                xFormat = xFormat + 'h'
            if flag & flagYShort:
                yFormat = yFormat + 'B'
            elif not flag & flagYsame:
                yFormat = yFormat + 'h'
            flags[j] = flag
            j = j + 1
        if j >= nCoordinates:
            break
    assert j == nCoordinates, 'bad glyph flags'
    xDataLen = struct.calcsize(xFormat)
    yDataLen = struct.calcsize(yFormat)
    if len(data) - pos - (xDataLen + yDataLen) >= 4:
        log.warning('too much glyph data: %d excess bytes', len(data) - pos - (xDataLen + yDataLen))
    xCoordinates = struct.unpack(xFormat, data[pos:pos + xDataLen])
    yCoordinates = struct.unpack(yFormat, data[pos + xDataLen:pos + xDataLen + yDataLen])
    return (flags, xCoordinates, yCoordinates)