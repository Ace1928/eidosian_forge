from fontTools.misc.fixedTools import fixedToFloat as fi2fl, floatToFixed as fl2fi
from fontTools.misc.textTools import tobytes, tostr
import struct
import re
def calcsize(fmt):
    formatstring, names, fixes = getformat(fmt)
    return struct.calcsize(formatstring)