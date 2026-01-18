from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
def mergeFeatures(lst):
    assert lst
    self = otTables.Feature()
    self.FeatureParams = None
    self.LookupListIndex = mergeLookupLists([l.LookupListIndex for l in lst if l.LookupListIndex])
    self.LookupCount = len(self.LookupListIndex)
    return self