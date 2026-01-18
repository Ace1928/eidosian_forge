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
def makeAnchor(data, klass=ot.Anchor):
    assert len(data) <= 2
    anchor = klass()
    anchor.Format = 1
    anchor.XCoordinate, anchor.YCoordinate = intSplitComma(data[0])
    if len(data) > 1 and data[1] != '':
        anchor.Format = 2
        anchor.AnchorPoint = int(data[1])
    return anchor