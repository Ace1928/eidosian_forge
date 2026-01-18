from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def get_room(self):
    """Maximum number of bytes that can be added to characteristic
        while still being beneficial to merge it into another one."""
    count = len(self.items)
    return max(0, (self.overhead - 1) // count - self.width)