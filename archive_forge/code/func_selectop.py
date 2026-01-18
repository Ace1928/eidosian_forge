from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
def selectop(table, field, value, op, complement=False):
    """Select rows where the function `op` applied to the given field and
    the given value returns `True`."""
    return select(table, field, lambda v: op(v, value), complement=complement)