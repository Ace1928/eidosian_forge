from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def read_longInt(self, b0, data, index):
    value, = struct.unpack('>l', data[index:index + 4])
    return (value, index + 4)