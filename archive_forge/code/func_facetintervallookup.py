from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def facetintervallookup(table, key, start='start', stop='stop', value=None, include_stop=False):
    """

    Construct a faceted interval lookup for the given table. E.g.::

        >>> import petl as etl
        >>> table = (('type', 'start', 'stop', 'value'),
        ...          ('apple', 1, 4, 'foo'),
        ...          ('apple', 3, 7, 'bar'),
        ...          ('orange', 4, 9, 'baz'))
        >>> lkp = etl.facetintervallookup(table, key='type', start='start', stop='stop')
        >>> lkp['apple'].search(1, 2)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(2, 4)
        [('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar')]
        >>> lkp['apple'].search(2, 5)
        [('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar')]
        >>> lkp['orange'].search(2, 5)
        [('orange', 4, 9, 'baz')]
        >>> lkp['orange'].search(9, 14)
        []
        >>> lkp['orange'].search(19, 140)
        []
        >>> lkp['apple'].search(1)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(2)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(4)
        [('apple', 3, 7, 'bar')]
        >>> lkp['apple'].search(5)
        [('apple', 3, 7, 'bar')]
        >>> lkp['orange'].search(5)
        [('orange', 4, 9, 'baz')]

    """
    trees = facettupletrees(table, key, start=start, stop=stop, value=value)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookup(trees[k], include_stop=include_stop)
    return out