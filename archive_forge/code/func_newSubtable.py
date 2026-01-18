from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
@staticmethod
def newSubtable(format):
    """Return a new instance of a subtable for the given format
        ."""
    subtableClass = CmapSubtable.getSubtableClass(format)
    return subtableClass(format)