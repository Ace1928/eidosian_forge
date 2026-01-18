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
def parseEncoding1(charset, file, haveSupplement, strings):
    nRanges = readCard8(file)
    encoding = ['.notdef'] * 256
    glyphID = 1
    for i in range(nRanges):
        code = readCard8(file)
        nLeft = readCard8(file)
        for glyphID in range(glyphID, glyphID + nLeft + 1):
            encoding[code] = charset[glyphID]
            code = code + 1
        glyphID = glyphID + 1
    return encoding