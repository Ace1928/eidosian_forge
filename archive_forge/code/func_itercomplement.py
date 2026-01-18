from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
def itercomplement(ta, tb, strict):
    ita = (tuple(row) for row in iter(ta))
    itb = (tuple(row) for row in iter(tb))
    ahdr = tuple(next(ita))
    next(itb)
    yield ahdr
    try:
        a = next(ita)
    except StopIteration:
        pass
    else:
        try:
            b = next(itb)
        except StopIteration:
            yield a
            for row in ita:
                yield row
        else:
            while True:
                if b is None or Comparable(a) < Comparable(b):
                    yield a
                    try:
                        a = next(ita)
                    except StopIteration:
                        break
                elif a == b:
                    try:
                        a = next(ita)
                    except StopIteration:
                        break
                    if not strict:
                        try:
                            b = next(itb)
                        except StopIteration:
                            b = None
                else:
                    try:
                        b = next(itb)
                    except StopIteration:
                        b = None