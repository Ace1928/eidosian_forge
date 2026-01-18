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
def instantiateAvar(varfont, axisLimits):
    segments = varfont['avar'].segments
    pinnedAxes = set(axisLimits.pinnedLocation())
    if pinnedAxes.issuperset(segments):
        log.info('Dropping avar table')
        del varfont['avar']
        return
    log.info('Instantiating avar table')
    for axis in pinnedAxes:
        if axis in segments:
            del segments[axis]
    normalizedRanges = axisLimits.normalize(varfont, usingAvar=False)
    newSegments = {}
    for axisTag, mapping in segments.items():
        if not _isValidAvarSegmentMap(axisTag, mapping):
            continue
        if mapping and axisTag in normalizedRanges:
            axisRange = normalizedRanges[axisTag]
            mappedMin = floatToFixedToFloat(piecewiseLinearMap(axisRange.minimum, mapping), 14)
            mappedDef = floatToFixedToFloat(piecewiseLinearMap(axisRange.default, mapping), 14)
            mappedMax = floatToFixedToFloat(piecewiseLinearMap(axisRange.maximum, mapping), 14)
            mappedAxisLimit = NormalizedAxisTripleAndDistances(mappedMin, mappedDef, mappedMax, axisRange.distanceNegative, axisRange.distancePositive)
            newMapping = {}
            for fromCoord, toCoord in mapping.items():
                if fromCoord < axisRange.minimum or fromCoord > axisRange.maximum:
                    continue
                fromCoord = axisRange.renormalizeValue(fromCoord)
                assert mappedMin <= toCoord <= mappedMax
                toCoord = mappedAxisLimit.renormalizeValue(toCoord)
                fromCoord = floatToFixedToFloat(fromCoord, 14)
                toCoord = floatToFixedToFloat(toCoord, 14)
                newMapping[fromCoord] = toCoord
            newMapping.update({-1.0: -1.0, 0.0: 0.0, 1.0: 1.0})
            newSegments[axisTag] = newMapping
        else:
            newSegments[axisTag] = mapping
    varfont['avar'].segments = newSegments