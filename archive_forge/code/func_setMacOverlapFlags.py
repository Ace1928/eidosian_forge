from fontTools.misc.fixedTools import (
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import _g_l_y_f
from fontTools import varLib
from fontTools import subset  # noqa: F401
from fontTools.varLib import builder
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.instancer import names
from .featureVars import instantiateFeatureVariations
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.varLib.instancer import solver
import collections
import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import logging
import os
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings
def setMacOverlapFlags(glyfTable):
    flagOverlapCompound = _g_l_y_f.OVERLAP_COMPOUND
    flagOverlapSimple = _g_l_y_f.flagOverlapSimple
    for glyphName in glyfTable.keys():
        glyph = glyfTable[glyphName]
        if glyph.isComposite():
            glyph.components[0].flags |= flagOverlapCompound
        elif glyph.numberOfContours > 0:
            glyph.flags[0] |= flagOverlapSimple