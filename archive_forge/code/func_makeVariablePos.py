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
def makeVariablePos(self, location, varscalar):
    if not self.varstorebuilder:
        raise FeatureLibError("Can't define a variable scalar in a non-variable font", location)
    varscalar.axes = self.axes
    if not varscalar.does_vary:
        return (varscalar.default, None)
    default, index = varscalar.add_to_variation_store(self.varstorebuilder, self.model_cache, self.font.get('avar'))
    device = None
    if index is not None and index != 4294967295:
        device = buildVarDevTable(index)
    return (default, device)