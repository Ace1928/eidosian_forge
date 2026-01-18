from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_hvcurveto(self, index):
    """dx1 dx2 dy2 dy3 {dya dxb dyb dxc dxd dxe dye dyf}* dxf?
        {dxa dxb dyb dyc dyd dxe dye dxf}+ dyf?
        """
    args = self.popall()
    while args:
        args = self.hcurveto(args)
        if args:
            args = self.vcurveto(args)