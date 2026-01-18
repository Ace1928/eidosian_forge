from fontTools.misc.timeTools import timestampNow
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from functools import reduce
import operator
import logging
def sumLists(lst):
    l = []
    for item in lst:
        l.extend(item)
    return l