from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def getCoordWidth(self):
    """Return 2 if coordinates are (x, y) as in gvar, 1 if single values
        as in cvar, or 0 if empty.
        """
    firstDelta = next((c for c in self.coordinates if c is not None), None)
    if firstDelta is None:
        return 0
    if type(firstDelta) in (int, float):
        return 1
    if type(firstDelta) is tuple and len(firstDelta) == 2:
        return 2
    raise TypeError('invalid type of delta; expected (int or float) number, or Tuple[number, number]: %r' % firstDelta)