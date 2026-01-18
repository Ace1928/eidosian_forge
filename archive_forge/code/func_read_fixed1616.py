from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def read_fixed1616(self, b0, data, index):
    value, = struct.unpack('>l', data[index:index + 4])
    return (fixedToFloat(value, precisionBits=16), index + 4)