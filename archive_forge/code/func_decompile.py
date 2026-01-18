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
def decompile(self, data, glyfTable):
    flags = struct.unpack('>H', data[:2])[0]
    self.flags = int(flags)
    data = data[2:]
    numAxes = int(data[0])
    data = data[1:]
    if flags & VarComponentFlags.GID_IS_24BIT:
        glyphID = int(struct.unpack('>L', b'\x00' + data[:3])[0])
        data = data[3:]
        flags ^= VarComponentFlags.GID_IS_24BIT
    else:
        glyphID = int(struct.unpack('>H', data[:2])[0])
        data = data[2:]
    self.glyphName = glyfTable.getGlyphName(int(glyphID))
    if flags & VarComponentFlags.AXIS_INDICES_ARE_SHORT:
        axisIndices = array.array('H', data[:2 * numAxes])
        if sys.byteorder != 'big':
            axisIndices.byteswap()
        data = data[2 * numAxes:]
        flags ^= VarComponentFlags.AXIS_INDICES_ARE_SHORT
    else:
        axisIndices = array.array('B', data[:numAxes])
        data = data[numAxes:]
    assert len(axisIndices) == numAxes
    axisIndices = list(axisIndices)
    axisValues = array.array('h', data[:2 * numAxes])
    if sys.byteorder != 'big':
        axisValues.byteswap()
    data = data[2 * numAxes:]
    assert len(axisValues) == numAxes
    axisValues = [fi2fl(v, 14) for v in axisValues]
    self.location = {glyfTable.axisTags[i]: v for i, v in zip(axisIndices, axisValues)}

    def read_transform_component(data, values):
        if flags & values.flag:
            return (data[2:], fi2fl(struct.unpack('>h', data[:2])[0], values.fractionalBits) * values.scale)
        else:
            return (data, values.defaultValue)
    for attr_name, mapping_values in VAR_COMPONENT_TRANSFORM_MAPPING.items():
        data, value = read_transform_component(data, mapping_values)
        setattr(self.transform, attr_name, value)
    if flags & VarComponentFlags.UNIFORM_SCALE:
        if flags & VarComponentFlags.HAVE_SCALE_X and (not flags & VarComponentFlags.HAVE_SCALE_Y):
            self.transform.scaleY = self.transform.scaleX
            flags |= VarComponentFlags.HAVE_SCALE_Y
        flags ^= VarComponentFlags.UNIFORM_SCALE
    return data