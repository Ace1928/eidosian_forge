from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
@add_method(otTables.LookupList)
def mapMarkFilteringSets(self, markFilteringSetMap):
    for l in self.Lookup:
        if not l:
            continue
        l.mapMarkFilteringSets(markFilteringSetMap)