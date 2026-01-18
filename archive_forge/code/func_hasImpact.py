from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def hasImpact(self):
    """Returns True if this TupleVariation has any visible impact.

        If the result is False, the TupleVariation can be omitted from the font
        without making any visible difference.
        """
    return any((c is not None for c in self.coordinates))