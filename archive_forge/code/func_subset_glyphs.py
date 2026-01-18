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
def subset_glyphs(self, s):
    s.glyphs = None
    tables_format12_bmp = []
    table_plat0_enc3 = {}
    table_plat3_enc1 = {}
    for t in self.tables:
        if t.platformID == 0 and t.platEncID == 3:
            table_plat0_enc3[t.language] = t
        if t.platformID == 3 and t.platEncID == 1:
            table_plat3_enc1[t.language] = t
        if t.format == 14:
            t.uvsDict = {v: [(u, g) for u, g in l if g in s.glyphs_requested or u in s.unicodes_requested] for v, l in t.uvsDict.items()}
            t.uvsDict = {v: l for v, l in t.uvsDict.items() if l}
        elif t.isUnicode():
            t.cmap = {u: g for u, g in t.cmap.items() if g in s.glyphs_requested or u in s.unicodes_requested}
            if t.format == 12 and t.cmap and (max(t.cmap.keys()) < 65536):
                tables_format12_bmp.append(t)
        else:
            t.cmap = {u: g for u, g in t.cmap.items() if g in s.glyphs_requested}
    for t in tables_format12_bmp:
        if t.platformID == 0 and t.platEncID == 4 and (t.language in table_plat0_enc3) and (table_plat0_enc3[t.language].cmap == t.cmap):
            t.cmap.clear()
        elif t.platformID == 3 and t.platEncID == 10 and (t.language in table_plat3_enc1) and (table_plat3_enc1[t.language].cmap == t.cmap):
            t.cmap.clear()
    self.tables = [t for t in self.tables if (t.cmap if t.format != 14 else t.uvsDict)]
    self.numSubTables = len(self.tables)
    return True