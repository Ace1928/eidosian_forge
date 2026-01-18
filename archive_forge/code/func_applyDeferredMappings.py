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
def applyDeferredMappings(self):
    for setter, sym, e in self._deferredMappings:
        log.debug("Applying deferred mapping for symbol '%s' %s", sym, type(e).__name__)
        try:
            mapped = self[sym]
        except KeyError:
            raise e
        setter(mapped)
        log.debug('Set to %s', mapped)
    self._deferredMappings = []