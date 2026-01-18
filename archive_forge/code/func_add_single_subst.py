from fontTools.misc import sstruct
from fontTools.misc.textTools import Tag, tostr, binary2num, safeEval
from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lookupDebugInfo import (
from fontTools.feaLib.parser import Parser
from fontTools.feaLib.ast import FeatureFile
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.otlLib import builder as otl
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.ttLib import newTable, getTableModule
from fontTools.ttLib.tables import otBase, otTables
from fontTools.otlLib.builder import (
from fontTools.otlLib.error import OpenTypeLibError
from fontTools.varLib.varStore import OnlineVarStoreBuilder
from fontTools.varLib.builder import buildVarDevTable
from fontTools.varLib.featureVars import addFeatureVariationsRaw
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from collections import defaultdict
import copy
import itertools
from io import StringIO
import logging
import warnings
import os
def add_single_subst(self, location, prefix, suffix, mapping, forceChain):
    if self.cur_feature_name_ == 'aalt':
        for from_glyph, to_glyph in mapping.items():
            alts = self.aalt_alternates_.setdefault(from_glyph, [])
            if to_glyph not in alts:
                alts.append(to_glyph)
        return
    if prefix or suffix or forceChain:
        self.add_single_subst_chained_(location, prefix, suffix, mapping)
        return
    lookup = self.get_lookup_(location, SingleSubstBuilder)
    for from_glyph, to_glyph in mapping.items():
        if from_glyph in lookup.mapping:
            if to_glyph == lookup.mapping[from_glyph]:
                log.info('Removing duplicate single substitution from glyph "%s" to "%s" at %s', from_glyph, to_glyph, location)
            else:
                raise FeatureLibError('Already defined rule for replacing glyph "%s" by "%s"' % (from_glyph, lookup.mapping[from_glyph]), location)
        lookup.mapping[from_glyph] = to_glyph