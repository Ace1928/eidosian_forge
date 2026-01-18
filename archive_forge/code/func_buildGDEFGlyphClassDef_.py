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
def buildGDEFGlyphClassDef_(self):
    if self.glyphClassDefs_:
        classes = {g: c for g, (c, _) in self.glyphClassDefs_.items()}
    else:
        classes = {}
        for lookup in self.lookups_:
            classes.update(lookup.inferGlyphClasses())
        for markClass in self.parseTree.markClasses.values():
            for markClassDef in markClass.definitions:
                for glyph in markClassDef.glyphSet():
                    classes[glyph] = 3
    if classes:
        result = otTables.GlyphClassDef()
        result.classDefs = classes
        return result
    else:
        return None