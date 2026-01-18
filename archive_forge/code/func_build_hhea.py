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
def build_hhea(self):
    if not self.hhea_:
        return
    table = self.font.get('hhea')
    if not table:
        table = self.font['hhea'] = newTable('hhea')
        table.decompile(b'\x00' * 36, self.font)
        table.tableVersion = 65536
    if 'caretoffset' in self.hhea_:
        table.caretOffset = self.hhea_['caretoffset']
    if 'ascender' in self.hhea_:
        table.ascent = self.hhea_['ascender']
    if 'descender' in self.hhea_:
        table.descent = self.hhea_['descender']
    if 'linegap' in self.hhea_:
        table.lineGap = self.hhea_['linegap']