from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_rrcurveto(self, index):
    """{dxa dya dxb dyb dxc dyc}+ rrcurveto"""
    args = self.popall()
    for i in range(0, len(args), 6):
        dxa, dya, dxb, dyb, dxc, dyc = args[i:i + 6]
        self.rCurveTo((dxa, dya), (dxb, dyb), (dxc, dyc))