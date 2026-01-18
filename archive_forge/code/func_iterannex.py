from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def iterannex(tables, missing):
    its = [iter(t) for t in tables]
    hdrs = []
    for it in its:
        try:
            hdrs.append(next(it))
        except StopIteration:
            hdrs.append([])
    outhdr = tuple(chain(*hdrs))
    yield outhdr
    for rows in izip_longest(*its):
        outrow = list()
        for i, row in enumerate(rows):
            lh = len(hdrs[i])
            if row is None:
                row = [missing] * len(hdrs[i])
            else:
                lr = len(row)
                if lr < lh:
                    row = list(row)
                    row.extend([missing] * (lh - lr))
                elif lr > lh:
                    row = row[:lh]
            outrow.extend(row)
        yield tuple(outrow)