from fontTools import config
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables
from fontTools.ttLib.tables.otBase import USE_HARFBUZZ_REPACKER
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.pens.basePen import NullPen
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.subset.util import _add_method, _uniq_sort
from fontTools.subset.cff import *
from fontTools.subset.svg import *
from fontTools.varLib import varStore  # for subset_varidxes
from fontTools.ttLib.tables._n_a_m_e import NameRecordVisitor
import sys
import struct
import array
import logging
from collections import Counter, defaultdict
from functools import reduce
from types import MethodType
@_add_method(ttLib.getTableClass('GSUB'), ttLib.getTableClass('GPOS'))
def prune_lookups(self, remap=True):
    """Remove (default) or neuter unreferenced lookups"""
    if self.table.ScriptList:
        feature_indices = self.table.ScriptList.collect_features()
    else:
        feature_indices = []
    if self.table.FeatureList:
        lookup_indices = self.table.FeatureList.collect_lookups(feature_indices)
    else:
        lookup_indices = []
    if getattr(self.table, 'FeatureVariations', None):
        lookup_indices += self.table.FeatureVariations.collect_lookups(feature_indices)
    lookup_indices = _uniq_sort(lookup_indices)
    if self.table.LookupList:
        lookup_indices = self.table.LookupList.closure_lookups(lookup_indices)
    else:
        lookup_indices = []
    if remap:
        self.subset_lookups(lookup_indices)
    else:
        self.neuter_lookups(lookup_indices)