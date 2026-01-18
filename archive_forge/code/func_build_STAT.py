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
def build_STAT(self):
    if not self.stat_:
        return
    axes = self.stat_.get('DesignAxes')
    if not axes:
        raise FeatureLibError('DesignAxes not defined', None)
    axisValueRecords = self.stat_.get('AxisValueRecords')
    axisValues = {}
    format4_locations = []
    for tag in axes:
        axisValues[tag.tag] = []
    if axisValueRecords is not None:
        for avr in axisValueRecords:
            valuesDict = {}
            if avr.flags > 0:
                valuesDict['flags'] = avr.flags
            if len(avr.locations) == 1:
                location = avr.locations[0]
                values = location.values
                if len(values) == 1:
                    valuesDict.update({'value': values[0], 'name': avr.names})
                if len(values) == 2:
                    valuesDict.update({'value': values[0], 'linkedValue': values[1], 'name': avr.names})
                if len(values) == 3:
                    nominal, minVal, maxVal = values
                    valuesDict.update({'nominalValue': nominal, 'rangeMinValue': minVal, 'rangeMaxValue': maxVal, 'name': avr.names})
                axisValues[location.tag].append(valuesDict)
            else:
                valuesDict.update({'location': {i.tag: i.values[0] for i in avr.locations}, 'name': avr.names})
                format4_locations.append(valuesDict)
    designAxes = [{'ordering': a.axisOrder, 'tag': a.tag, 'name': a.names, 'values': axisValues[a.tag]} for a in axes]
    nameTable = self.font.get('name')
    if not nameTable:
        nameTable = self.font['name'] = newTable('name')
        nameTable.names = []
    if 'ElidedFallbackNameID' in self.stat_:
        nameID = self.stat_['ElidedFallbackNameID']
        name = nameTable.getDebugName(nameID)
        if not name:
            raise FeatureLibError(f'ElidedFallbackNameID {nameID} points to a nameID that does not exist in the "name" table', None)
    elif 'ElidedFallbackName' in self.stat_:
        nameID = self.stat_['ElidedFallbackName']
    otl.buildStatTable(self.font, designAxes, locations=format4_locations, elidedFallbackName=nameID)