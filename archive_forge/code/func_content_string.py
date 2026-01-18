from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
from . import grUtils
from array import array
from functools import reduce
import struct, re, sys
def content_string(contents):
    res = ''
    for element in contents:
        if isinstance(element, tuple):
            continue
        res += element
    return res.strip()