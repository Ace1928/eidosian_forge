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
def buildFormat8(self, writer, font, values):
    minGlyphID, maxGlyphID = (values[0][0], values[-1][0])
    if len(values) != maxGlyphID - minGlyphID + 1:
        return None
    valueSize = self.converter.staticSize
    return (6 + len(values) * valueSize, 8, lambda: self.writeFormat8(writer, font, values))