from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
def iterintersection(a, b):
    ita = iter(a)
    itb = iter(b)
    ahdr = next(ita)
    next(itb)
    yield tuple(ahdr)
    try:
        a = tuple(next(ita))
        b = tuple(next(itb))
        while True:
            if Comparable(a) < Comparable(b):
                a = tuple(next(ita))
            elif a == b:
                yield a
                a = tuple(next(ita))
                b = tuple(next(itb))
            else:
                b = tuple(next(itb))
    except StopIteration:
        pass