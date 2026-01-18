from __future__ import absolute_import, print_function, division
import itertools
import operator
from collections import OrderedDict
from petl.compat import next, string_types, reduce, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, rowgroupby
from petl.util.base import values
from petl.util.counting import nrows
from petl.transform.sorts import sort, mergesort
from petl.transform.basics import cut
from petl.transform.dedup import distinct
def groupselectmax(table, key, value, presorted=False, buffersize=None, tempdir=None, cache=True):
    """Group by the `key` field then return the row with the maximum of the
    `value` field within each group. N.B., will only return one row for each
    group, even if multiple rows have the same (maximum) value."""
    return groupselectfirst(sort(table, value, reverse=True), key, presorted=presorted, buffersize=buffersize, tempdir=tempdir, cache=cache)