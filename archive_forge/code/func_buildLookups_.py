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
def buildLookups_(self, tag):
    assert tag in ('GPOS', 'GSUB'), tag
    for lookup in self.lookups_:
        lookup.lookup_index = None
    lookups = []
    for lookup in self.lookups_:
        if lookup.table != tag:
            continue
        lookup.lookup_index = len(lookups)
        self.lookup_locations[tag][str(lookup.lookup_index)] = LookupDebugInfo(location=str(lookup.location), name=self.get_lookup_name_(lookup), feature=None)
        lookups.append(lookup)
    otLookups = []
    for l in lookups:
        try:
            otLookups.append(l.build())
        except OpenTypeLibError as e:
            raise FeatureLibError(str(e), e.location) from e
        except Exception as e:
            location = self.lookup_locations[tag][str(l.lookup_index)].location
            raise FeatureLibError(str(e), location) from e
    return otLookups