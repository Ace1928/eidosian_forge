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
def add_single_pos(self, location, prefix, suffix, pos, forceChain):
    if prefix or suffix or forceChain:
        self.add_single_pos_chained_(location, prefix, suffix, pos)
    else:
        lookup = self.get_lookup_(location, SinglePosBuilder)
        for glyphs, value in pos:
            if not glyphs:
                raise FeatureLibError('Empty glyph class in positioning rule', location)
            otValueRecord = self.makeOpenTypeValueRecord(location, value, pairPosContext=False)
            for glyph in glyphs:
                try:
                    lookup.add_pos(location, glyph, otValueRecord)
                except OpenTypeLibError as e:
                    raise FeatureLibError(str(e), e.location) from e