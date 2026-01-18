from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
def selectrangeopenright(table, field, minv, maxv, complement=False):
    """Select rows where the given field is greater than `minv` and
    less than or equal to `maxv`."""
    minv = Comparable(minv)
    maxv = Comparable(maxv)
    return select(table, field, lambda v: minv < v <= maxv, complement=complement)