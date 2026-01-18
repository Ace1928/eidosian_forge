from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_seac(self, index):
    """asb adx ady bchar achar seac"""
    from fontTools.encodings.StandardEncoding import StandardEncoding
    asb, adx, ady, bchar, achar = self.popall()
    baseGlyph = StandardEncoding[bchar]
    self.pen.addComponent(baseGlyph, (1, 0, 0, 1, 0, 0))
    accentGlyph = StandardEncoding[achar]
    adx = adx + self.sbx - asb
    self.pen.addComponent(accentGlyph, (1, 0, 0, 1, adx, ady))