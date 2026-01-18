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
def storeVariationIndices(self, varIdxes) -> int:
    key = tuple(varIdxes)
    varIndexBase = self.varIndexCache.get(key)
    if varIndexBase is None:
        for i in range(len(self.varIdxes) - len(varIdxes) + 1):
            if self.varIdxes[i:i + len(varIdxes)] == varIdxes:
                self.varIndexCache[key] = varIndexBase = i
                break
    if varIndexBase is None:
        for n in range(len(varIdxes) - 1, 0, -1):
            if self.varIdxes[-n:] == varIdxes[:n]:
                varIndexBase = len(self.varIdxes) - n
                self.varIndexCache[key] = varIndexBase
                self.varIdxes.extend(varIdxes[n:])
                break
    if varIndexBase is None:
        self.varIndexCache[key] = varIndexBase = len(self.varIdxes)
        self.varIdxes.extend(varIdxes)
    return varIndexBase