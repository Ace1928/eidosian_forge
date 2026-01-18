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
def flush_(self):
    if self.classDef1_ is None or self.classDef2_ is None:
        return
    st = buildPairPosClassesSubtable(self.values_, self.builder_.glyphMap)
    if st.Coverage is None:
        return
    self.subtables_.append(st)
    self.forceSubtableBreak_ = False