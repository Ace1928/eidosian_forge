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
def canAdd(self, glyphs):
    if isinstance(glyphs, (set, frozenset)):
        glyphs = sorted(glyphs)
    glyphs = tuple(glyphs)
    if glyphs in self.classes_:
        return True
    for glyph in glyphs:
        if glyph in self.glyphs_:
            return False
    return True