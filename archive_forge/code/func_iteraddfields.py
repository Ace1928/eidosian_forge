from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def iteraddfields(source, field_defs):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    outhdr = list(hdr)
    value_indexes = []
    for fdef in field_defs:
        if len(fdef) == 2:
            name, value = fdef
            index = len(outhdr)
        else:
            name, value, index = fdef
        outhdr.insert(index, name)
        value_indexes.append((value, index))
    yield tuple(outhdr)
    for row in it:
        outrow = list(row)
        for value, index in value_indexes:
            if callable(value):
                row = Record(row, flds)
                v = value(row)
                outrow.insert(index, v)
            else:
                outrow.insert(index, value)
        yield tuple(outrow)