from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def arg_blend_number(self, name):
    out = []
    blendArgs = self.pop()
    numMasters = len(blendArgs)
    out.append(blendArgs)
    out.append('blend')
    dummy = self.popall()
    return blendArgs