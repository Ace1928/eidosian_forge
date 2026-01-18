import os
import copy
import enum
from operator import ior
import logging
from fontTools.colorLib.builder import MAX_PAINT_COLR_LAYER_COUNT, LayerReuseCache
from fontTools.misc import classifyTools
from fontTools.misc.roundTools import otRound
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables import otBase as otBase
from fontTools.ttLib.tables.otConverters import BaseFixedValue
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.models import nonNone, allNone, allEqual, allEqualTo, subList
from fontTools.varLib.varStore import VarStoreInstancer
from functools import reduce
from fontTools.otlLib.builder import buildSinglePos
from fontTools.otlLib.optimize.gpos import (
from .errors import (
def mergeVariableAttrs(self, out, lst, attrs) -> int:
    varIndexBase = ot.NO_VARIATION_INDEX
    varIdxes = []
    for attr in attrs:
        baseValue, varIdx = self.storeMastersForAttr(out, lst, attr)
        setattr(out, attr, baseValue)
        varIdxes.append(varIdx)
    if any((v != ot.NO_VARIATION_INDEX for v in varIdxes)):
        varIndexBase = self.storeVariationIndices(varIdxes)
    return varIndexBase