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
def add_specific_pair_pos(self, location, glyph1, value1, glyph2, value2):
    if not glyph1 or not glyph2:
        raise FeatureLibError('Empty glyph class in positioning rule', location)
    lookup = self.get_lookup_(location, PairPosBuilder)
    v1 = self.makeOpenTypeValueRecord(location, value1, pairPosContext=True)
    v2 = self.makeOpenTypeValueRecord(location, value2, pairPosContext=True)
    lookup.addGlyphPair(location, glyph1, v1, glyph2, v2)