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
def add_ligature_subst(self, location, prefix, glyphs, suffix, replacement, forceChain):
    if prefix or suffix or forceChain:
        chain = self.get_lookup_(location, ChainContextSubstBuilder)
        lookup = self.get_chained_lookup_(location, LigatureSubstBuilder)
        chain.rules.append(ChainContextualRule(prefix, glyphs, suffix, [lookup]))
    else:
        lookup = self.get_lookup_(location, LigatureSubstBuilder)
    if not all(glyphs):
        raise FeatureLibError('Empty glyph class in substitution', location)
    for g in itertools.product(*glyphs):
        lookup.ligatures[g] = replacement