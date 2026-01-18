from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def setUnicodeRanges(self, bits):
    """Set the 'ulUnicodeRange*' fields to the specified 'bits'."""
    ul1, ul2, ul3, ul4 = (0, 0, 0, 0)
    for bit in bits:
        if 0 <= bit < 32:
            ul1 |= 1 << bit
        elif 32 <= bit < 64:
            ul2 |= 1 << bit - 32
        elif 64 <= bit < 96:
            ul3 |= 1 << bit - 64
        elif 96 <= bit < 123:
            ul4 |= 1 << bit - 96
        else:
            raise ValueError('expected 0 <= int <= 122, found: %r' % bit)
    self.ulUnicodeRange1, self.ulUnicodeRange2 = (ul1, ul2)
    self.ulUnicodeRange3, self.ulUnicodeRange4 = (ul3, ul4)