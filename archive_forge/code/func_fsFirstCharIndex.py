from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
@fsFirstCharIndex.setter
def fsFirstCharIndex(self, value):
    self.usFirstCharIndex = value