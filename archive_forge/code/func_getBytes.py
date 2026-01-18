from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def getBytes(self, index, nBytes):
    if self.bytecode is not None:
        newIndex = index + nBytes
        bytes = self.bytecode[index:newIndex]
        index = newIndex
    else:
        bytes = self.program[index]
        index = index + 1
    assert len(bytes) == nBytes
    return (bytes, index)