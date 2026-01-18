from fontTools.ttLib import getSearchRange
from fontTools.misc.textTools import safeEval, readHex
from fontTools.misc.fixedTools import fixedToFloat as fi2fl, floatToFixed as fl2fi
from . import DefaultTable
import struct
import sys
import array
import logging
def getkern(self, format):
    for subtable in self.kernTables:
        if subtable.format == format:
            return subtable
    return None