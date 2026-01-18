from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def alternatingLineto(self, isHorizontal):
    args = self.popall()
    for arg in args:
        if isHorizontal:
            point = (arg, 0)
        else:
            point = (0, arg)
        self.rLineTo(point)
        isHorizontal = not isHorizontal