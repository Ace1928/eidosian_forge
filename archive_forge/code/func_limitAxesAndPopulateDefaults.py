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
def limitAxesAndPopulateDefaults(self, varfont) -> 'AxisLimits':
    """Return a new AxisLimits with defaults filled in from fvar table.

        If all axis limits already have defaults, return self.
        """
    fvar = varfont['fvar']
    fvarTriples = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in fvar.axes}
    newLimits = {}
    for axisTag, triple in self.items():
        fvarTriple = fvarTriples[axisTag]
        default = fvarTriple[1]
        if triple is None:
            newLimits[axisTag] = AxisTriple(default, default, default)
        else:
            newLimits[axisTag] = triple.limitRangeAndPopulateDefaults(fvarTriple)
    return type(self)(newLimits)