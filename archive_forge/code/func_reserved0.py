from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from fontTools.misc.fixedTools import (
from . import DefaultTable
import math
@reserved0.setter
def reserved0(self, value):
    self.caretOffset = value