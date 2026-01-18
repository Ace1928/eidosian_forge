from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def iteraddfieldusingcontext(table, field, query):
    it = iter(table)
    try:
        hdr = tuple(next(it))
    except StopIteration:
        hdr = ()
    flds = list(map(text_type, hdr))
    yield (hdr + (field,))
    flds.append(field)
    it = (Record(row, flds) for row in it)
    prv = None
    try:
        cur = next(it)
    except StopIteration:
        return
    for nxt in it:
        v = query(prv, cur, nxt)
        yield (tuple(cur) + (v,))
        prv = Record(tuple(cur) + (v,), flds)
        cur = nxt
    v = query(prv, cur, None)
    yield (tuple(cur) + (v,))