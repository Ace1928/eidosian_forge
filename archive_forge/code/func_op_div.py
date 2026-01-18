from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_div(self, index):
    num2 = self.pop()
    num1 = self.pop()
    d1 = num1 // num2
    d2 = num1 / num2
    if d1 == d2:
        self.push(d1)
    else:
        self.push(d2)