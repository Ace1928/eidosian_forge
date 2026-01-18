from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_rlineto(self, index):
    args = self.popall()
    for i in range(0, len(args), 2):
        point = args[i:i + 2]
        self.rLineTo(point)