from fontTools import ttLib
from fontTools.ttLib.tables._c_m_a_p import cmap_classes
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import ValueRecord, valueRecordFormatDict
from fontTools.otlLib import builder as otl
from contextlib import contextmanager
from fontTools.ttLib import newTable
from fontTools.feaLib.lookupDebugInfo import LOOKUP_DEBUG_ENV_VAR, LOOKUP_DEBUG_INFO_KEY
from operator import setitem
import os
import logging
def parseFeatureList(lines, lookupMap=None, featureMap=None):
    self = ot.FeatureList()
    self.FeatureRecord = []
    with lines.between('feature table'):
        for line in lines:
            name, featureTag, lookups = line
            if featureMap is not None:
                assert name not in featureMap, 'Duplicate feature name: %s' % name
                featureMap[name] = len(self.FeatureRecord)
            try:
                assert int(name) == len(self.FeatureRecord), '%d %d' % (name, len(self.FeatureRecord))
            except ValueError:
                pass
            featureRec = ot.FeatureRecord()
            featureRec.FeatureTag = featureTag
            featureRec.Feature = ot.Feature()
            self.FeatureRecord.append(featureRec)
            feature = featureRec.Feature
            feature.FeatureParams = None
            syms = stripSplitComma(lookups)
            feature.LookupListIndex = theList = [None] * len(syms)
            for i, sym in enumerate(syms):
                setReference(mapLookup, lookupMap, sym, setitem, theList, i)
            feature.LookupCount = len(feature.LookupListIndex)
    self.FeatureCount = len(self.FeatureRecord)
    return self