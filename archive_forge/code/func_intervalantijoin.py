from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def intervalantijoin(left, right, lstart='start', lstop='stop', rstart='start', rstop='stop', lkey=None, rkey=None, include_stop=False, missing=None):
    """
    Return rows from the `left` table with no overlapping rows from the `right`
    table.

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """
    assert (lkey is None) == (rkey is None), 'facet key field must be provided for both or neither table'
    return IntervalAntiJoinView(left, right, lstart=lstart, lstop=lstop, rstart=rstart, rstop=rstop, lkey=lkey, rkey=rkey, include_stop=include_stop, missing=missing)