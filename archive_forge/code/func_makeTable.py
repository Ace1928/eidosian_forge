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
def makeTable(self, tag):
    table = getattr(otTables, tag, None)()
    table.Version = 65536
    table.ScriptList = otTables.ScriptList()
    table.ScriptList.ScriptRecord = []
    table.FeatureList = otTables.FeatureList()
    table.FeatureList.FeatureRecord = []
    table.LookupList = otTables.LookupList()
    table.LookupList.Lookup = self.buildLookups_(tag)
    feature_indices = {}
    required_feature_indices = {}
    scripts = {}
    sortFeatureTag = lambda f: (f[0][2], f[0][1], f[0][0], f[1])
    for key, lookups in sorted(self.features_.items(), key=sortFeatureTag):
        script, lang, feature_tag = key
        lookup_indices = tuple([l.lookup_index for l in lookups if l.lookup_index is not None])
        size_feature = tag == 'GPOS' and feature_tag == 'size'
        force_feature = self.any_feature_variations(feature_tag, tag)
        if len(lookup_indices) == 0 and (not size_feature) and (not force_feature):
            continue
        for ix in lookup_indices:
            try:
                self.lookup_locations[tag][str(ix)] = self.lookup_locations[tag][str(ix)]._replace(feature=key)
            except KeyError:
                warnings.warn('feaLib.Builder subclass needs upgrading to stash debug information. See fonttools#2065.')
        feature_key = (feature_tag, lookup_indices)
        feature_index = feature_indices.get(feature_key)
        if feature_index is None:
            feature_index = len(table.FeatureList.FeatureRecord)
            frec = otTables.FeatureRecord()
            frec.FeatureTag = feature_tag
            frec.Feature = otTables.Feature()
            frec.Feature.FeatureParams = self.buildFeatureParams(feature_tag)
            frec.Feature.LookupListIndex = list(lookup_indices)
            frec.Feature.LookupCount = len(lookup_indices)
            table.FeatureList.FeatureRecord.append(frec)
            feature_indices[feature_key] = feature_index
        scripts.setdefault(script, {}).setdefault(lang, []).append(feature_index)
        if self.required_features_.get((script, lang)) == feature_tag:
            required_feature_indices[script, lang] = feature_index
    for script, lang_features in sorted(scripts.items()):
        srec = otTables.ScriptRecord()
        srec.ScriptTag = script
        srec.Script = otTables.Script()
        srec.Script.DefaultLangSys = None
        srec.Script.LangSysRecord = []
        for lang, feature_indices in sorted(lang_features.items()):
            langrec = otTables.LangSysRecord()
            langrec.LangSys = otTables.LangSys()
            langrec.LangSys.LookupOrder = None
            req_feature_index = required_feature_indices.get((script, lang))
            if req_feature_index is None:
                langrec.LangSys.ReqFeatureIndex = 65535
            else:
                langrec.LangSys.ReqFeatureIndex = req_feature_index
            langrec.LangSys.FeatureIndex = [i for i in feature_indices if i != req_feature_index]
            langrec.LangSys.FeatureCount = len(langrec.LangSys.FeatureIndex)
            if lang == 'dflt':
                srec.Script.DefaultLangSys = langrec.LangSys
            else:
                langrec.LangSysTag = lang
                srec.Script.LangSysRecord.append(langrec)
        srec.Script.LangSysCount = len(srec.Script.LangSysRecord)
        table.ScriptList.ScriptRecord.append(srec)
    table.ScriptList.ScriptCount = len(table.ScriptList.ScriptRecord)
    table.FeatureList.FeatureCount = len(table.FeatureList.FeatureRecord)
    table.LookupList.LookupCount = len(table.LookupList.Lookup)
    return table