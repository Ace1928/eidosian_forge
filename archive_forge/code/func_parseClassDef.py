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
def parseClassDef(lines, font, klass=ot.ClassDef):
    classDefs = {}
    with lines.between('class definition'):
        for line in lines:
            glyph = makeGlyph(line[0])
            assert glyph not in classDefs, glyph
            classDefs[glyph] = int(line[1])
    return makeClassDef(classDefs, font, klass)