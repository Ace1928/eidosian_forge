from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def itercut(source, spec, missing=None):
    it = iter(source)
    spec = tuple(spec)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    indices = asindices(hdr, spec)
    transform = rowgetter(*indices)
    yield transform(hdr)
    for row in it:
        try:
            yield transform(row)
        except IndexError:
            yield tuple((row[i] if i < len(row) else missing for i in indices))