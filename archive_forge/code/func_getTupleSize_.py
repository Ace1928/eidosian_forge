from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
@staticmethod
def getTupleSize_(flags, axisCount):
    size = 4
    if flags & EMBEDDED_PEAK_TUPLE != 0:
        size += axisCount * 2
    if flags & INTERMEDIATE_REGION != 0:
        size += axisCount * 4
    return size