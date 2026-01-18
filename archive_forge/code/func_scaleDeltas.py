from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def scaleDeltas(self, scalar):
    if scalar == 1.0:
        return
    coordWidth = self.getCoordWidth()
    self.coordinates = [None if d is None else d * scalar if coordWidth == 1 else (d[0] * scalar, d[1] * scalar) for d in self.coordinates]