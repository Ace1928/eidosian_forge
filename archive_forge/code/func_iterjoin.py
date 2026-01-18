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
def iterjoin(left, right, lkey, rkey, leftouter=False, rightouter=False, missing=None, lprefix=None, rprefix=None):
    lit = iter(left)
    rit = iter(right)
    lhdr = next(lit)
    rhdr = next(rit)
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)
    lgetk = comparable_itemgetter(*lkind)
    rgetk = comparable_itemgetter(*rkind)
    rvind = [i for i in range(len(rhdr)) if i not in rkind]
    rgetv = rowgetter(*rvind)
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [text_type(lprefix) + text_type(f) for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([text_type(rprefix) + text_type(f) for f in rgetv(rhdr)])
    yield tuple(outhdr)

    def joinrows(_lrowgrp, _rrowgrp):
        if _rrowgrp is None:
            for lrow in _lrowgrp:
                outrow = list(lrow)
                outrow.extend([missing] * len(rvind))
                yield tuple(outrow)
        elif _lrowgrp is None:
            for rrow in _rrowgrp:
                outrow = [missing] * len(lhdr)
                for li, ri in zip(lkind, rkind):
                    outrow[li] = rrow[ri]
                outrow.extend(rgetv(rrow))
                yield tuple(outrow)
        else:
            _rrowgrp = list(_rrowgrp)
            for lrow in _lrowgrp:
                for rrow in _rrowgrp:
                    outrow = list(lrow)
                    outrow.extend(rgetv(rrow))
                    yield tuple(outrow)
    lgit = itertools.groupby(lit, key=lgetk)
    rgit = itertools.groupby(rit, key=rgetk)
    lrowgrp = []
    rrowgrp = []
    lkval, rkval = (Comparable(None), Comparable(None))
    try:
        lkval, lrowgrp = next(lgit)
        rkval, rrowgrp = next(rgit)
        while True:
            if lkval < rkval:
                if leftouter:
                    for row in joinrows(lrowgrp, None):
                        yield tuple(row)
                lkval, lrowgrp = next(lgit)
            elif lkval > rkval:
                if rightouter:
                    for row in joinrows(None, rrowgrp):
                        yield tuple(row)
                rkval, rrowgrp = next(rgit)
            else:
                for row in joinrows(lrowgrp, rrowgrp):
                    yield tuple(row)
                lkval, lrowgrp = next(lgit)
                rkval, rrowgrp = next(rgit)
    except StopIteration:
        pass
    if leftouter:
        if lkval > rkval:
            for row in joinrows(lrowgrp, None):
                yield tuple(row)
        for lkval, lrowgrp in lgit:
            for row in joinrows(lrowgrp, None):
                yield tuple(row)
    if rightouter:
        if lkval < rkval:
            for row in joinrows(None, rrowgrp):
                yield tuple(row)
        for rkval, rrowgrp in rgit:
            for row in joinrows(None, rrowgrp):
                yield tuple(row)