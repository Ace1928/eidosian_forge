from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
def layoutPreMerge(font):
    GDEF = font.get('GDEF')
    GSUB = font.get('GSUB')
    GPOS = font.get('GPOS')
    for t in [GSUB, GPOS]:
        if not t:
            continue
        if t.table.LookupList:
            lookupMap = {i: v for i, v in enumerate(t.table.LookupList.Lookup)}
            t.table.LookupList.mapLookups(lookupMap)
            t.table.FeatureList.mapLookups(lookupMap)
            if GDEF and GDEF.table.Version >= 65538 and GDEF.table.MarkGlyphSetsDef:
                markFilteringSetMap = {i: v for i, v in enumerate(GDEF.table.MarkGlyphSetsDef.Coverage)}
                t.table.LookupList.mapMarkFilteringSets(markFilteringSetMap)
        if t.table.FeatureList and t.table.ScriptList:
            featureMap = {i: v for i, v in enumerate(t.table.FeatureList.FeatureRecord)}
            t.table.ScriptList.mapFeatures(featureMap)