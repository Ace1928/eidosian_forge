import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
@cache
def split_edge(self, v1, v2):
    v1 = self.V[v1]
    v2 = self.V[v2]
    v1.disconnect(v2)
    try:
        vct = (v2.x_a - v1.x_a) / 2.0 + v1.x_a
    except TypeError:
        vct = (v2.x_a - v1.x_a) / decimal.Decimal(2.0) + v1.x_a
    vc = self.V[tuple(vct)]
    vc.connect(v1)
    vc.connect(v2)
    return vc