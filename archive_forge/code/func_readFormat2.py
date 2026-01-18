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
def readFormat2(self, reader, font):
    mapping = {}
    pos = reader.pos - 2
    unitSize, numUnits = (reader.readUShort(), reader.readUShort())
    assert unitSize >= 4 + self.converter.staticSize, unitSize
    for i in range(numUnits):
        reader.seek(pos + i * unitSize + 12)
        last = reader.readUShort()
        first = reader.readUShort()
        value = self.converter.read(reader, font, tableDict=None)
        if last != 65535:
            for k in range(first, last + 1):
                mapping[font.getGlyphName(k)] = value
    return mapping