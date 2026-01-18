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
def limitRangeAndPopulateDefaults(self, fvarTriple) -> 'AxisTriple':
    """Return a new AxisTriple with the default value filled in.

        Set default to fvar axis default if the latter is within the min/max range,
        otherwise set default to the min or max value, whichever is closer to the
        fvar axis default.
        If the default value is already set, return self.
        """
    minimum = self.minimum
    if minimum is None:
        minimum = fvarTriple[0]
    default = self.default
    if default is None:
        default = fvarTriple[1]
    maximum = self.maximum
    if maximum is None:
        maximum = fvarTriple[2]
    minimum = max(minimum, fvarTriple[0])
    maximum = max(maximum, fvarTriple[0])
    minimum = min(minimum, fvarTriple[2])
    maximum = min(maximum, fvarTriple[2])
    default = max(minimum, min(maximum, default))
    return AxisTriple(minimum, default, maximum)