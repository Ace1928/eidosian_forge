from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import safeEval
from functools import partial
from . import DefaultTable
from . import grUtils
import struct
def compileAttributes12(self, attrs, fmt):
    data = b''
    for e in grUtils.entries(attrs):
        data += sstruct.pack(fmt, {'attNum': e[0], 'num': e[1]}) + struct.pack('>%dh' % len(e[2]), *e[2])
    return data