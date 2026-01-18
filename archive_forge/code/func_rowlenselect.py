from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
def rowlenselect(table, n, complement=False):
    """Select rows of length `n`."""
    where = lambda row: len(row) == n
    return select(table, where, complement=complement)