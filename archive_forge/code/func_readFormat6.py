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
def readFormat6(self, reader, font):
    mapping = {}
    pos = reader.pos - 2
    unitSize = reader.readUShort()
    assert unitSize >= 2 + self.converter.staticSize, unitSize
    for i in range(reader.readUShort()):
        reader.seek(pos + i * unitSize + 12)
        glyphID = reader.readUShort()
        value = self.converter.read(reader, font, tableDict=None)
        if glyphID != 65535:
            mapping[font.getGlyphName(glyphID)] = value
    return mapping