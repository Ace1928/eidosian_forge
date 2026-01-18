from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def intervalrecordlookupone(table, start='start', stop='stop', include_stop=False, strict=True):
    """
    As :func:`petl.transform.intervals.intervallookupone` but return records
    instead of tuples.

    """
    tree = recordtree(table, start=start, stop=stop)
    return IntervalTreeLookupOne(tree, include_stop=include_stop, strict=strict)