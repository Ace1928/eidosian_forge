from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_vhcurveto(self, index):
    """dy1 dx2 dy2 dx3 {dxa dxb dyb dyc dyd dxe dye dxf}* dyf? vhcurveto (30)
        {dya dxb dyb dxc dxd dxe dye dyf}+ dxf? vhcurveto
        """
    args = self.popall()
    while args:
        args = self.vcurveto(args)
        if args:
            args = self.hcurveto(args)