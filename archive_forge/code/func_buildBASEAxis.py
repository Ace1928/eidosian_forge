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
def buildBASEAxis(self, axis):
    if not axis:
        return
    bases, scripts = axis
    axis = otTables.Axis()
    axis.BaseTagList = otTables.BaseTagList()
    axis.BaseTagList.BaselineTag = bases
    axis.BaseTagList.BaseTagCount = len(bases)
    axis.BaseScriptList = otTables.BaseScriptList()
    axis.BaseScriptList.BaseScriptRecord = []
    axis.BaseScriptList.BaseScriptCount = len(scripts)
    for script in sorted(scripts):
        record = otTables.BaseScriptRecord()
        record.BaseScriptTag = script[0]
        record.BaseScript = otTables.BaseScript()
        record.BaseScript.BaseLangSysCount = 0
        record.BaseScript.BaseValues = otTables.BaseValues()
        record.BaseScript.BaseValues.DefaultIndex = bases.index(script[1])
        record.BaseScript.BaseValues.BaseCoord = []
        record.BaseScript.BaseValues.BaseCoordCount = len(script[2])
        for c in script[2]:
            coord = otTables.BaseCoord()
            coord.Format = 1
            coord.Coordinate = c
            record.BaseScript.BaseValues.BaseCoord.append(coord)
        axis.BaseScriptList.BaseScriptRecord.append(record)
    return axis