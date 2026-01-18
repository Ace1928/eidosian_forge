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
def makeBaseRecords(data, coverage, c, classCount):
    records = []
    idx = {}
    for glyph in coverage.glyphs:
        idx[glyph] = len(records)
        record = c.BaseRecordClass()
        anchors = [None] * classCount
        setattr(record, c.BaseAnchor, anchors)
        records.append(record)
    for (glyph, klass), anchor in data.items():
        record = records[idx[glyph]]
        anchors = getattr(record, c.BaseAnchor)
        assert anchors[klass] is None, (glyph, klass)
        anchors[klass] = anchor
    return records