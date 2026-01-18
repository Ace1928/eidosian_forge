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
def storeMastersForAttr(self, out, lst, attr):
    master_values = [getattr(item, attr) for item in lst]
    is_fixed_size_float = False
    conv = out.getConverterByName(attr)
    if isinstance(conv, BaseFixedValue):
        is_fixed_size_float = True
        master_values = [conv.toInt(v) for v in master_values]
    baseValue = master_values[0]
    varIdx = ot.NO_VARIATION_INDEX
    if not allEqual(master_values):
        baseValue, varIdx = self.store_builder.storeMasters(master_values)
    if is_fixed_size_float:
        baseValue = conv.fromInt(baseValue)
    return (baseValue, varIdx)