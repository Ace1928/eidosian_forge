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
def setInputCoverage_(self, glyphs, subtable):
    subtable.InputGlyphCount = len(glyphs)
    subtable.InputCoverage = []
    for g in glyphs:
        coverage = buildCoverage(g, self.glyphMap)
        subtable.InputCoverage.append(coverage)