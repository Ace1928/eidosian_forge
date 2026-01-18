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
def buildGDEF(self):
    gdef = otTables.GDEF()
    gdef.GlyphClassDef = self.buildGDEFGlyphClassDef_()
    gdef.AttachList = otl.buildAttachList(self.attachPoints_, self.glyphMap)
    gdef.LigCaretList = otl.buildLigCaretList(self.ligCaretCoords_, self.ligCaretPoints_, self.glyphMap)
    gdef.MarkAttachClassDef = self.buildGDEFMarkAttachClassDef_()
    gdef.MarkGlyphSetsDef = self.buildGDEFMarkGlyphSetsDef_()
    gdef.Version = 65538 if gdef.MarkGlyphSetsDef else 65536
    if self.varstorebuilder:
        store = self.varstorebuilder.finish()
        if store:
            gdef.Version = 65539
            gdef.VarStore = store
            varidx_map = store.optimize()
            gdef.remap_device_varidxes(varidx_map)
            if 'GPOS' in self.font:
                self.font['GPOS'].table.remap_device_varidxes(varidx_map)
        self.model_cache.clear()
    if any((gdef.GlyphClassDef, gdef.AttachList, gdef.LigCaretList, gdef.MarkAttachClassDef, gdef.MarkGlyphSetsDef)) or hasattr(gdef, 'VarStore'):
        result = newTable('GDEF')
        result.table = gdef
        return result
    else:
        return None