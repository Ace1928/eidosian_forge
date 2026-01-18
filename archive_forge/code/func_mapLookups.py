from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
@add_method(otTables.FeatureList)
def mapLookups(self, lookupMap):
    for f in self.FeatureRecord:
        if not f or not f.Feature:
            continue
        f.Feature.mapLookups(lookupMap)