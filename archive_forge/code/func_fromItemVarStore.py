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
@classmethod
def fromItemVarStore(cls, itemVarStore, fvarAxes):
    axisOrder = [axis.axisTag for axis in fvarAxes]
    regions = [region.get_support(fvarAxes) for region in itemVarStore.VarRegionList.Region]
    tupleVarData = []
    itemCounts = []
    for varData in itemVarStore.VarData:
        variations = []
        varDataRegions = (regions[i] for i in varData.VarRegionIndex)
        for axes, coordinates in zip(varDataRegions, zip(*varData.Item)):
            variations.append(TupleVariation(axes, list(coordinates)))
        tupleVarData.append(variations)
        itemCounts.append(varData.ItemCount)
    return cls(regions, axisOrder, tupleVarData, itemCounts)