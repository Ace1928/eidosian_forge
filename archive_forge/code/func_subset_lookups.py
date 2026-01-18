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
def subset_lookups(self, lookup_indices):
    """Retains specified lookups, then removes empty features, language
    systems, and scripts."""
    if self.table.LookupList:
        self.table.LookupList.subset_lookups(lookup_indices)
    if self.table.FeatureList:
        feature_indices = self.table.FeatureList.subset_lookups(lookup_indices)
    else:
        feature_indices = []
    if getattr(self.table, 'FeatureVariations', None):
        feature_indices += self.table.FeatureVariations.subset_lookups(lookup_indices)
    feature_indices = _uniq_sort(feature_indices)
    if self.table.FeatureList:
        self.table.FeatureList.subset_features(feature_indices)
    if getattr(self.table, 'FeatureVariations', None):
        self.table.FeatureVariations.subset_features(feature_indices)
    if self.table.ScriptList:
        self.table.ScriptList.subset_features(feature_indices, self.retain_empty_scripts())