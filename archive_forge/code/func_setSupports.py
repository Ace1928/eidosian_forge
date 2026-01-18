from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def setSupports(self, supports):
    self._model = None
    self._supports = list(supports)
    if not self._supports[0]:
        del self._supports[0]
    self._cache = {}
    self._data = None