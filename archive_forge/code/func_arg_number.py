from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def arg_number(self, name):
    if isinstance(self.stack[0], list):
        out = self.arg_blend_number(self.stack)
    else:
        out = self.pop()
    return out