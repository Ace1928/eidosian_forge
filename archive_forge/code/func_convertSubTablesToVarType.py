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
@classmethod
def convertSubTablesToVarType(cls, table):
    for path in dfs_base_table(table, skip_root=True, predicate=lambda path: getattr(type(path[-1].value), 'VarType', None) is not None):
        st = path[-1]
        subTable = st.value
        varType = type(subTable).VarType
        newSubTable = varType()
        newSubTable.__dict__.update(subTable.__dict__)
        newSubTable.populateDefaults()
        parent = path[-2].value
        if st.index is not None:
            getattr(parent, st.name)[st.index] = newSubTable
        else:
            setattr(parent, st.name, newSubTable)