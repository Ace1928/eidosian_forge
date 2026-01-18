from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_vsindex(self, index):
    vi = self.pop()
    self.vsIndex = vi
    self.numRegions = self.private.getNumRegions(vi)