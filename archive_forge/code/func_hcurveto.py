from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def hcurveto(self, args):
    dxa, dxb, dyb, dyc = args[:4]
    args = args[4:]
    if len(args) == 1:
        dxc = args[0]
        args = []
    else:
        dxc = 0
    self.rCurveTo((dxa, 0), (dxb, dyb), (dxc, dyc))
    return args