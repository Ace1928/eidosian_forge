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
def mergeThings(self, out, lst):
    masterModel = None
    origTTFs = None
    if None in lst:
        if allNone(lst):
            if out is not None:
                raise FoundANone(self, got=lst)
            return
        origTTFs = self.ttfs
        if self.ttfs:
            self.ttfs = subList([v is not None for v in lst], self.ttfs)
        masterModel = self.model
        model, lst = masterModel.getSubModel(lst)
        self.setModel(model)
    super(VariationMerger, self).mergeThings(out, lst)
    if masterModel:
        self.setModel(masterModel)
    if origTTFs:
        self.ttfs = origTTFs