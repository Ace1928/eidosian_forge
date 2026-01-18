from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def read_realNumber(self, b0, data, index):
    number = ''
    while True:
        b = byteord(data[index])
        index = index + 1
        nibble0 = (b & 240) >> 4
        nibble1 = b & 15
        if nibble0 == 15:
            break
        number = number + realNibbles[nibble0]
        if nibble1 == 15:
            break
        number = number + realNibbles[nibble1]
    return (float(number), index)