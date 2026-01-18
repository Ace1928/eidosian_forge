from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
def hashintersection(a, b):
    """
    Alternative implementation of
    :func:`petl.transform.setops.intersection`, where the intersection
    is executed by constructing an in-memory set for all rows found in the
    right hand table, then iterating over rows from the left hand table.

    May be faster and/or more resource efficient where the right table is small
    and the left table is large.

    """
    return HashIntersectionView(a, b)