from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from fontTools.misc.fixedTools import (
from . import DefaultTable
import math
@descender.setter
def descender(self, value):
    self.descent = value