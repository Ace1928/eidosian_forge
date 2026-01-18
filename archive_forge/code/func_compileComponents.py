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
def compileComponents(self, glyfTable):
    data = b''
    lastcomponent = len(self.components) - 1
    more = 1
    haveInstructions = 0
    for i in range(len(self.components)):
        if i == lastcomponent:
            haveInstructions = hasattr(self, 'program')
            more = 0
        compo = self.components[i]
        data = data + compo.compile(more, haveInstructions, glyfTable)
    if haveInstructions:
        instructions = self.program.getBytecode()
        data = data + struct.pack('>h', len(instructions)) + instructions
    return data