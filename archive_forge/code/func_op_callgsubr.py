from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_callgsubr(self, index):
    subrIndex = self.pop()
    subr = self.globalSubrs[subrIndex + self.globalBias]
    self.execute(subr)