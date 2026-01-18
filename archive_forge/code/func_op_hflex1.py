from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_hflex1(self, index):
    dx1, dy1, dx2, dy2, dx3, dx4, dx5, dy5, dx6 = self.popall()
    dy3 = dy4 = 0
    dy6 = -(dy1 + dy2 + dy3 + dy4 + dy5)
    self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
    self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))