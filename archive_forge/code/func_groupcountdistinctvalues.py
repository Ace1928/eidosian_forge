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
def groupcountdistinctvalues(table, key, value):
    """Group by the `key` field then count the number of distinct values in the
    `value` field."""
    s1 = cut(table, key, value)
    s2 = distinct(s1)
    s3 = aggregate(s2, key, len)
    return s3