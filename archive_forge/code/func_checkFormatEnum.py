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
def checkFormatEnum(self, out, lst, validate=lambda _: True):
    fmt = out.Format
    formatEnum = out.formatEnum
    ok = False
    try:
        fmt = formatEnum(fmt)
    except ValueError:
        pass
    else:
        ok = validate(fmt)
    if not ok:
        raise UnsupportedFormat(self, subtable=type(out).__name__, value=fmt)
    expected = fmt
    got = []
    for v in lst:
        fmt = getattr(v, 'Format', None)
        try:
            fmt = formatEnum(fmt)
        except ValueError:
            pass
        got.append(fmt)
    if not allEqualTo(expected, got):
        raise InconsistentFormats(self, subtable=type(out).__name__, expected=expected, got=got)
    return expected