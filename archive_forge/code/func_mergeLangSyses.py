from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
def mergeLangSyses(lst):
    assert lst
    assert all((l.ReqFeatureIndex == 65535 for l in lst))
    self = otTables.LangSys()
    self.LookupOrder = None
    self.ReqFeatureIndex = 65535
    self.FeatureIndex = mergeFeatureLists([l.FeatureIndex for l in lst if l.FeatureIndex])
    self.FeatureCount = len(self.FeatureIndex)
    return self