from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def iterstack(sources, missing, trim, pad):
    its = [iter(t) for t in sources]
    hdrs = []
    for it in its:
        try:
            hdrs.append(next(it))
        except StopIteration:
            hdrs.append([])
    hdr = hdrs[0]
    n = len(hdr)
    yield tuple(hdr)
    for it in its:
        for row in it:
            outrow = tuple(row)
            if trim:
                outrow = outrow[:n]
            if pad and len(outrow) < n:
                outrow += (missing,) * (n - len(outrow))
            yield outrow