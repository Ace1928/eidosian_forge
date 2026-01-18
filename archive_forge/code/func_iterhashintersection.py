from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
def iterhashintersection(a, b):
    ita = iter(a)
    ahdr = next(ita)
    yield tuple(ahdr)
    itb = iter(b)
    next(itb)
    bcnt = Counter((tuple(row) for row in itb))
    for ar in ita:
        t = tuple(ar)
        if bcnt[t] > 0:
            yield t
            bcnt[t] -= 1