from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def read_operator(self, b0, data, index):
    if b0 == 12:
        op = (b0, byteord(data[index]))
        index = index + 1
    else:
        op = b0
    try:
        operator = self.operators[op]
    except KeyError:
        return (None, index)
    value = self.handle_operator(operator)
    return (value, index)