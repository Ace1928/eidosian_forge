from fontTools import ttLib
import fontTools.merge.base
from fontTools.merge.cmap import (
from fontTools.merge.layout import layoutPreMerge, layoutPostMerge
from fontTools.merge.options import Options
import fontTools.merge.tables
from fontTools.misc.loggingTools import Timer
from functools import reduce
import sys
import logging
def mergeObjects(self, returnTable, logic, tables):
    allKeys = set.union(set(), *(vars(table).keys() for table in tables if table is not NotImplemented))
    for key in allKeys:
        log.info(' %s', key)
        try:
            mergeLogic = logic[key]
        except KeyError:
            try:
                mergeLogic = logic['*']
            except KeyError:
                raise Exception("Don't know how to merge key %s of class %s" % (key, returnTable.__class__.__name__))
        if mergeLogic is NotImplemented:
            continue
        value = mergeLogic((getattr(table, key, NotImplemented) for table in tables))
        if value is not NotImplemented:
            setattr(returnTable, key, value)
    return returnTable