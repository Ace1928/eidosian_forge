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
@_add_method(otTables.Anchor)
def prune_hints(self):
    if self.Format == 2:
        self.Format = 1
    elif self.Format == 3:
        for name in ('XDeviceTable', 'YDeviceTable'):
            v = getattr(self, name, None)
            if v is not None and v.is_hinting():
                setattr(self, name, None)
        if self.XDeviceTable is None and self.YDeviceTable is None:
            self.Format = 1