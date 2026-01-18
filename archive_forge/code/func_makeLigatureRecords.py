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
def makeLigatureRecords(data, coverage, c, classCount):
    records = [None] * len(coverage.glyphs)
    idx = {g: i for i, g in enumerate(coverage.glyphs)}
    for (glyph, klass, compIdx, compCount), anchor in data.items():
        record = records[idx[glyph]]
        if record is None:
            record = records[idx[glyph]] = ot.LigatureAttach()
            record.ComponentCount = compCount
            record.ComponentRecord = [ot.ComponentRecord() for i in range(compCount)]
            for compRec in record.ComponentRecord:
                compRec.LigatureAnchor = [None] * classCount
        assert record.ComponentCount == compCount, (glyph, record.ComponentCount, compCount)
        anchors = record.ComponentRecord[compIdx - 1].LigatureAnchor
        assert anchors[klass] is None, (glyph, compIdx, klass)
        anchors[klass] = anchor
    return records