from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def storeMasters(self, master_values, *, round=round):
    deltas = self._model.getDeltas(master_values, round=round)
    base = deltas.pop(0)
    return (base, self.storeDeltas(deltas, round=noRound))