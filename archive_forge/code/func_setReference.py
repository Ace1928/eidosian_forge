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
def setReference(mapper, mapping, sym, setter, collection, key):
    try:
        mapped = mapper(sym, mapping)
    except ReferenceNotFoundError as e:
        try:
            if mapping is not None:
                mapping.addDeferredMapping(lambda ref: setter(collection, key, ref), sym, e)
                return
        except AttributeError:
            pass
        raise
    setter(collection, key, mapped)