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
def packCharset0(charset, isCID, strings):
    fmt = 0
    data = [packCard8(fmt)]
    if isCID:
        getNameID = getCIDfromName
    else:
        getNameID = getSIDfromName
    for name in charset[1:]:
        data.append(packCard16(getNameID(name, strings)))
    return bytesjoin(data)