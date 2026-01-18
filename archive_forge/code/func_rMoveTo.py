from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def rMoveTo(self, point):
    self.pen.moveTo(self._nextPoint(point))
    self.sawMoveTo = 1