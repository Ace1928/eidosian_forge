from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_rlinecurve(self, index):
    """{dxa dya}+ dxb dyb dxc dyc dxd dyd rlinecurve"""
    args = self.popall()
    lineArgs = args[:-6]
    for i in range(0, len(lineArgs), 2):
        self.rLineTo(lineArgs[i:i + 2])
    dxb, dyb, dxc, dyc, dxd, dyd = args[-6:]
    self.rCurveTo((dxb, dyb), (dxc, dyc), (dxd, dyd))