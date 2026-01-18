from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def getUsedPoints(self):
    if None not in self.coordinates:
        return frozenset()
    used = frozenset([i for i, p in enumerate(self.coordinates) if p is not None])
    return used if used else None