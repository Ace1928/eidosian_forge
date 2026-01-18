from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
def cvtFromUVS(val):
    assert 0 <= val < 16777216
    fourByteString = struct.pack('>L', val)
    return fourByteString[1:]