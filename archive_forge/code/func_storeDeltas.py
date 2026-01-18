from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def storeDeltas(self, deltas, *, round=round):
    deltas = [round(d) for d in deltas]
    if len(deltas) == len(self._supports) + 1:
        deltas = tuple(deltas[1:])
    else:
        assert len(deltas) == len(self._supports)
        deltas = tuple(deltas)
    varIdx = self._cache.get(deltas)
    if varIdx is not None:
        return varIdx
    if not self._data:
        self._add_VarData()
    inner = len(self._data.Item)
    if inner == 65535:
        self._add_VarData()
        return self.storeDeltas(deltas)
    self._data.addItem(deltas, round=noRound)
    varIdx = (self._outer << 16) + inner
    self._cache[deltas] = varIdx
    return varIdx