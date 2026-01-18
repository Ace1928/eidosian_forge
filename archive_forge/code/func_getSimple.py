from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
from . import grUtils
from array import array
from functools import reduce
import struct, re, sys
def getSimple(self, attrs, *attr_list):
    for k in attr_list:
        if k in attrs:
            setattr(self, k, int(safeEval(attrs[k])))