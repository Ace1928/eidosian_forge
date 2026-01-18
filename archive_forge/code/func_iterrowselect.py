from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
def iterrowselect(source, where, missing, complement):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    yield tuple(hdr)
    it = (Record(row, flds, missing=missing) for row in it)
    for row in it:
        if bool(where(row)) != complement:
            yield tuple(row)