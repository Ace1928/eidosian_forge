from fontTools import config
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables
from fontTools.ttLib.tables.otBase import USE_HARFBUZZ_REPACKER
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.pens.basePen import NullPen
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.subset.util import _add_method, _uniq_sort
from fontTools.subset.cff import *
from fontTools.subset.svg import *
from fontTools.varLib import varStore  # for subset_varidxes
from fontTools.ttLib.tables._n_a_m_e import NameRecordVisitor
import sys
import struct
import array
import logging
from collections import Counter, defaultdict
from functools import reduce
from types import MethodType
@_add_method(ttLib.getTableClass('GSUB'), ttLib.getTableClass('GPOS'))
def subset_script_tags(self, tags):
    langsys = {}
    script_tags = set()
    for tag in tags:
        script_tag, lang_tag = tag.split('.') if '.' in tag else (tag, '*')
        script_tags.add(script_tag.ljust(4))
        langsys.setdefault(script_tag, set()).add(lang_tag.ljust(4))
    if self.table.ScriptList:
        self.table.ScriptList.ScriptRecord = [s for s in self.table.ScriptList.ScriptRecord if s.ScriptTag in script_tags]
        self.table.ScriptList.ScriptCount = len(self.table.ScriptList.ScriptRecord)
        for record in self.table.ScriptList.ScriptRecord:
            if record.ScriptTag in langsys and '*   ' not in langsys[record.ScriptTag]:
                record.Script.LangSysRecord = [l for l in record.Script.LangSysRecord if l.LangSysTag in langsys[record.ScriptTag]]
                record.Script.LangSysCount = len(record.Script.LangSysRecord)
                if 'dflt' not in langsys[record.ScriptTag]:
                    record.Script.DefaultLangSys = None