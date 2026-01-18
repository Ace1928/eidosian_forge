from fontTools import ttLib
from fontTools.ttLib.tables._c_m_a_p import cmap_classes
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import ValueRecord, valueRecordFormatDict
from fontTools.otlLib import builder as otl
from contextlib import contextmanager
from fontTools.ttLib import newTable
from fontTools.feaLib.lookupDebugInfo import LOOKUP_DEBUG_ENV_VAR, LOOKUP_DEBUG_INFO_KEY
from operator import setitem
import os
import logging
def parseSinglePos(lines, font, _lookupMap=None):
    values = {}
    for line in lines:
        assert len(line) == 3, line
        w = line[0].title().replace(' ', '')
        assert w in valueRecordFormatDict
        g = makeGlyph(line[1])
        v = int(line[2])
        if g not in values:
            values[g] = ValueRecord()
        assert not hasattr(values[g], w), (g, w)
        setattr(values[g], w, v)
    return otl.buildSinglePosSubtable(values, font.getReverseGlyphMap())