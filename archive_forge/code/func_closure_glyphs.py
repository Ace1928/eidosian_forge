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
@_add_method(ttLib.getTableClass('cmap'))
def closure_glyphs(self, s):
    tables = [t for t in self.tables if t.isUnicode()]
    for table in tables:
        if table.format == 14:
            for cmap in table.uvsDict.values():
                glyphs = {g for u, g in cmap if u in s.unicodes_requested}
                if None in glyphs:
                    glyphs.remove(None)
                s.glyphs.update(glyphs)
        else:
            cmap = table.cmap
            intersection = s.unicodes_requested.intersection(cmap.keys())
            s.glyphs.update((cmap[u] for u in intersection))
    s.unicodes_missing = s.unicodes_requested.copy()
    for table in tables:
        s.unicodes_missing.difference_update(table.cmap)