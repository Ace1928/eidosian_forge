from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
def mergeLookupLists(lst):
    return sumLists(lst)