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
def instantiateFvar(varfont, axisLimits):
    location = axisLimits.pinnedLocation()
    fvar = varfont['fvar']
    if set(location).issuperset((axis.axisTag for axis in fvar.axes)):
        log.info('Dropping fvar table')
        del varfont['fvar']
        return
    log.info('Instantiating fvar table')
    axes = []
    for axis in fvar.axes:
        axisTag = axis.axisTag
        if axisTag in location:
            continue
        if axisTag in axisLimits:
            triple = axisLimits[axisTag]
            if triple.default is None:
                triple = (triple.minimum, axis.defaultValue, triple.maximum)
            axis.minValue, axis.defaultValue, axis.maxValue = triple
        axes.append(axis)
    fvar.axes = axes
    instances = []
    for instance in fvar.instances:
        if any((instance.coordinates[axis] != value for axis, value in location.items())):
            continue
        for axisTag in location:
            del instance.coordinates[axisTag]
        if not isInstanceWithinAxisRanges(instance.coordinates, axisLimits):
            continue
        instances.append(instance)
    fvar.instances = instances