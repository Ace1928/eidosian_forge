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
def decompileComponents(self, data, glyfTable):
    self.components = []
    more = 1
    haveInstructions = 0
    while more:
        component = GlyphComponent()
        more, haveInstr, data = component.decompile(data, glyfTable)
        haveInstructions = haveInstructions | haveInstr
        self.components.append(component)
    if haveInstructions:
        numInstructions, = struct.unpack('>h', data[:2])
        data = data[2:]
        self.program = ttProgram.Program()
        self.program.fromBytecode(data[:numInstructions])
        data = data[numInstructions:]
        if len(data) >= 4:
            log.warning('too much glyph data at the end of composite glyph: %d excess bytes', len(data))