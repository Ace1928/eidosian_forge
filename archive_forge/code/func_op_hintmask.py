from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_hintmask(self, index):
    if not self.hintMaskBytes:
        self.countHints()
        self.hintMaskBytes = (self.hintCount + 7) // 8
    hintMaskBytes, index = self.callingStack[-1].getBytes(index, self.hintMaskBytes)
    return (hintMaskBytes, index)