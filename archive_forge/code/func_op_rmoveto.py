from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_rmoveto(self, index):
    if self.flexing:
        return
    self.endPath()
    self.rMoveTo(self.popall())