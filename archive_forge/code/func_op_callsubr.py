from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_callsubr(self, index):
    subrIndex = self.pop()
    subr = self.subrs[subrIndex]
    self.execute(subr)