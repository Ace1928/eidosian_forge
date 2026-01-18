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
@_add_method(ttLib.getTableModule('glyf').Glyph)
def remapComponentsFast(self, glyphidmap):
    if not self.data or struct.unpack('>h', self.data[:2])[0] >= 0:
        return
    data = self.data = bytearray(self.data)
    i = 10
    more = 1
    while more:
        flags = data[i] << 8 | data[i + 1]
        glyphID = data[i + 2] << 8 | data[i + 3]
        glyphID = glyphidmap[glyphID]
        data[i + 2] = glyphID >> 8
        data[i + 3] = glyphID & 255
        i += 4
        flags = int(flags)
        if flags & 1:
            i += 4
        else:
            i += 2
        if flags & 8:
            i += 2
        elif flags & 64:
            i += 4
        elif flags & 128:
            i += 8
        more = flags & 32