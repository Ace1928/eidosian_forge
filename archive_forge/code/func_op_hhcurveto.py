from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_hhcurveto(self, index):
    """dy1? {dxa dxb dyb dxc}+ hhcurveto"""
    args = self.popall()
    if len(args) % 2:
        dy1 = args[0]
        args = args[1:]
    else:
        dy1 = 0
    for i in range(0, len(args), 4):
        dxa, dxb, dyb, dxc = args[i:i + 4]
        self.rCurveTo((dxa, dy1), (dxb, dyb), (dxc, 0))
        dy1 = 0