from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def rLineTo(self, point):
    if not self.sawMoveTo:
        self.rMoveTo((0, 0))
    self.pen.lineTo(self._nextPoint(point))