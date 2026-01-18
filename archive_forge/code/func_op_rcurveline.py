from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_rcurveline(self, index):
    """{dxa dya dxb dyb dxc dyc}+ dxd dyd rcurveline"""
    args = self.popall()
    for i in range(0, len(args) - 2, 6):
        dxb, dyb, dxc, dyc, dxd, dyd = args[i:i + 6]
        self.rCurveTo((dxb, dyb), (dxc, dyc), (dxd, dyd))
    self.rLineTo(args[-2:])