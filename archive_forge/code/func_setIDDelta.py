from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
def setIDDelta(self, subHeader):
    subHeader.idDelta = 0
    minGI = subHeader.glyphIndexArray[0]
    for gid in subHeader.glyphIndexArray:
        if gid != 0 and gid < minGI:
            minGI = gid
    if minGI > 1:
        if minGI > 32767:
            subHeader.idDelta = -(65536 - minGI) - 1
        else:
            subHeader.idDelta = minGI - 1
        idDelta = subHeader.idDelta
        for i in range(subHeader.entryCount):
            gid = subHeader.glyphIndexArray[i]
            if gid > 0:
                subHeader.glyphIndexArray[i] = gid - idDelta