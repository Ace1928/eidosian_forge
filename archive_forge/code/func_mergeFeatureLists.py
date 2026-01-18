from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
def mergeFeatureLists(lst):
    d = {}
    for l in lst:
        for f in l:
            tag = f.FeatureTag
            if tag not in d:
                d[tag] = []
            d[tag].append(f.Feature)
    ret = []
    for tag in sorted(d.keys()):
        rec = otTables.FeatureRecord()
        rec.FeatureTag = tag
        rec.Feature = mergeFeatures(d[tag])
        ret.append(rec)
    return ret