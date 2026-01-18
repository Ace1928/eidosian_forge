from fontTools import ttLib
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables import otTables
from fontTools.merge.base import add_method, mergeObjects
from fontTools.merge.util import *
import logging
def mergeScripts(lst):
    assert lst
    if len(lst) == 1:
        return lst[0]
    langSyses = {}
    for sr in lst:
        for lsr in sr.LangSysRecord:
            if lsr.LangSysTag not in langSyses:
                langSyses[lsr.LangSysTag] = []
            langSyses[lsr.LangSysTag].append(lsr.LangSys)
    lsrecords = []
    for tag, langSys_list in sorted(langSyses.items()):
        lsr = otTables.LangSysRecord()
        lsr.LangSys = mergeLangSyses(langSys_list)
        lsr.LangSysTag = tag
        lsrecords.append(lsr)
    self = otTables.Script()
    self.LangSysRecord = lsrecords
    self.LangSysCount = len(lsrecords)
    dfltLangSyses = [s.DefaultLangSys for s in lst if s.DefaultLangSys]
    if dfltLangSyses:
        self.DefaultLangSys = mergeLangSyses(dfltLangSyses)
    else:
        self.DefaultLangSys = None
    return self