from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
def iterfieldselect(source, field, where, complement, missing):
    it = iter(source)
    try:
        hdr = next(it)
        yield tuple(hdr)
    except StopIteration:
        hdr = []
    indices = asindices(hdr, field)
    getv = operator.itemgetter(*indices)
    for row in it:
        try:
            v = getv(row)
        except IndexError:
            v = missing
        if bool(where(v)) != complement:
            yield tuple(row)