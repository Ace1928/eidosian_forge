from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
def mergeScriptRecords(lst):
    d = {}
    for l in lst:
        for s in l:
            tag = s.ScriptTag
            if tag not in d:
                d[tag] = []
            d[tag].append(s.Script)
    ret = []
    for tag in sorted(d.keys()):
        rec = otTables.ScriptRecord()
        rec.ScriptTag = tag
        rec.Script = mergeScripts(d[tag])
        ret.append(rec)
    return ret