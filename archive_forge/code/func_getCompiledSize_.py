from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def getCompiledSize_(self, subtables):
    if not subtables:
        return 0
    table = self.buildLookup_(copy.deepcopy(subtables))
    w = OTTableWriter()
    table.compile(w, self.font)
    size = len(w.getAllData())
    return size