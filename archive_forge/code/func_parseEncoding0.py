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
def parseEncoding0(charset, file, haveSupplement, strings):
    nCodes = readCard8(file)
    encoding = ['.notdef'] * 256
    for glyphID in range(1, nCodes + 1):
        code = readCard8(file)
        if code != 0:
            encoding[code] = charset[glyphID]
    return encoding