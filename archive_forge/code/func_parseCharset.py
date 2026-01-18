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
def parseCharset(numGlyphs, file, strings, isCID, fmt):
    charset = ['.notdef']
    count = 1
    if fmt == 1:
        nLeftFunc = readCard8
    else:
        nLeftFunc = readCard16
    while count < numGlyphs:
        first = readCard16(file)
        nLeft = nLeftFunc(file)
        if isCID:
            for CID in range(first, first + nLeft + 1):
                charset.append('cid' + str(CID).zfill(5))
        else:
            for SID in range(first, first + nLeft + 1):
                charset.append(strings[SID])
        count = count + nLeft + 1
    return charset