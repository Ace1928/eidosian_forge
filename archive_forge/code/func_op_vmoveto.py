from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_vmoveto(self, index):
    if self.flexing:
        self.push(0)
        self.exch()
        return
    self.endPath()
    self.rMoveTo((0, self.popall()[0]))