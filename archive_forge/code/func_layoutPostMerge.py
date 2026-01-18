from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
def layoutPostMerge(font):
    GDEF = font.get('GDEF')
    GSUB = font.get('GSUB')
    GPOS = font.get('GPOS')
    for t in [GSUB, GPOS]:
        if not t:
            continue
        if t.table.FeatureList and t.table.ScriptList:
            featureMap = GregariousIdentityDict(t.table.FeatureList.FeatureRecord)
            t.table.ScriptList.mapFeatures(featureMap)
            featureMap = AttendanceRecordingIdentityDict(t.table.FeatureList.FeatureRecord)
            t.table.ScriptList.mapFeatures(featureMap)
            usedIndices = featureMap.s
            t.table.FeatureList.FeatureRecord = [f for i, f in enumerate(t.table.FeatureList.FeatureRecord) if i in usedIndices]
            featureMap = NonhashableDict(t.table.FeatureList.FeatureRecord)
            t.table.ScriptList.mapFeatures(featureMap)
            t.table.FeatureList.FeatureCount = len(t.table.FeatureList.FeatureRecord)
        if t.table.LookupList:
            lookupMap = GregariousIdentityDict(t.table.LookupList.Lookup)
            t.table.FeatureList.mapLookups(lookupMap)
            t.table.LookupList.mapLookups(lookupMap)
            lookupMap = AttendanceRecordingIdentityDict(t.table.LookupList.Lookup)
            t.table.FeatureList.mapLookups(lookupMap)
            t.table.LookupList.mapLookups(lookupMap)
            usedIndices = lookupMap.s
            t.table.LookupList.Lookup = [l for i, l in enumerate(t.table.LookupList.Lookup) if i in usedIndices]
            lookupMap = NonhashableDict(t.table.LookupList.Lookup)
            t.table.FeatureList.mapLookups(lookupMap)
            t.table.LookupList.mapLookups(lookupMap)
            t.table.LookupList.LookupCount = len(t.table.LookupList.Lookup)
            if GDEF and GDEF.table.Version >= 65538:
                markFilteringSetMap = NonhashableDict(GDEF.table.MarkGlyphSetsDef.Coverage)
                t.table.LookupList.mapMarkFilteringSets(markFilteringSetMap)