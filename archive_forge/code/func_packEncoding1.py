from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
def packEncoding1(charset, encoding, strings):
    fmt = 1
    m = {}
    for code in range(len(encoding)):
        name = encoding[code]
        if name != '.notdef':
            m[name] = code
    ranges = []
    first = None
    end = 0
    for name in charset[1:]:
        code = m.get(name, -1)
        if first is None:
            first = code
        elif end + 1 != code:
            nLeft = end - first
            ranges.append((first, nLeft))
            first = code
        end = code
    nLeft = end - first
    ranges.append((first, nLeft))
    while ranges and ranges[-1][0] == -1:
        ranges.pop()
    data = [packCard8(fmt), packCard8(len(ranges))]
    for first, nLeft in ranges:
        if first == -1:
            first = 0
        data.append(packCard8(first) + packCard8(nLeft))
    return bytesjoin(data)