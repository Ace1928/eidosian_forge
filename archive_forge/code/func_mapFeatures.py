from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
@add_method(otTables.ScriptList)
def mapFeatures(self, featureMap):
    for s in self.ScriptRecord:
        if not s or not s.Script:
            continue
        s.Script.mapFeatures(featureMap)