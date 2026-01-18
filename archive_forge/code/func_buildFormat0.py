from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
def buildFormat0(self, writer, font, values):
    numGlyphs = len(font.getGlyphOrder())
    if len(values) != numGlyphs:
        return None
    valueSize = self.converter.staticSize
    return (2 + numGlyphs * valueSize, 0, lambda: self.writeFormat0(writer, font, values))