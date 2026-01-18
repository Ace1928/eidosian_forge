from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def rCurveTo(self, pt1, pt2, pt3):
    if not self.sawMoveTo:
        self.rMoveTo((0, 0))
    nextPoint = self._nextPoint
    self.pen.curveTo(nextPoint(pt1), nextPoint(pt2), nextPoint(pt3))