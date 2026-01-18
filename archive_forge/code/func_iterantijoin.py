from __future__ import absolute_import, print_function, division
import itertools
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.comparison import comparable_itemgetter, Comparable
from petl.util.base import Table, asindices, rowgetter, rowgroupby, \
from petl.transform.sorts import sort
from petl.transform.basics import cut, cutout
from petl.transform.dedup import distinct
def iterantijoin(left, right, lkey, rkey):
    lit = iter(left)
    rit = iter(right)
    lhdr = next(lit)
    rhdr = next(rit)
    yield tuple(lhdr)
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)
    lgetk = comparable_itemgetter(*lkind)
    rgetk = comparable_itemgetter(*rkind)
    lgit = itertools.groupby(lit, key=lgetk)
    rgit = itertools.groupby(rit, key=rgetk)
    lrowgrp = []
    lkval, rkval = (Comparable(None), Comparable(None))
    try:
        lkval, lrowgrp = next(lgit)
        rkval, _ = next(rgit)
        while True:
            if lkval < rkval:
                for row in lrowgrp:
                    yield tuple(row)
                lkval, lrowgrp = next(lgit)
            elif lkval > rkval:
                rkval, _ = next(rgit)
            else:
                lkval, lrowgrp = next(lgit)
                rkval, _ = next(rgit)
    except StopIteration:
        pass
    if lkval > rkval:
        for row in lrowgrp:
            yield tuple(row)
    for lkval, lrowgrp in lgit:
        for row in lrowgrp:
            yield tuple(row)