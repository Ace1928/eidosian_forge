from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def recalcCodePageRanges(self, ttFont, pruneOnly=False):
    unicodes = set()
    for table in ttFont['cmap'].tables:
        if table.isUnicode():
            unicodes.update(table.cmap.keys())
    bits = calcCodePageRanges(unicodes)
    if pruneOnly:
        bits &= self.getCodePageRanges()
    if not bits:
        bits = {0}
    self.setCodePageRanges(bits)
    return bits