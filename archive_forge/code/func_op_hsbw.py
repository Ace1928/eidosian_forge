from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_hsbw(self, index):
    sbx, wx = self.popall()
    self.width = wx
    self.sbx = sbx
    self.currentPoint = (sbx, self.currentPoint[1])