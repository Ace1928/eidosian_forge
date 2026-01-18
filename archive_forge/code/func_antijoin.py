from __future__ import absolute_import, print_function, division
import itertools
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.comparison import comparable_itemgetter, Comparable
from petl.util.base import Table, asindices, rowgetter, rowgroupby, \
from petl.transform.sorts import sort
from petl.transform.basics import cut, cutout
from petl.transform.dedup import distinct
def antijoin(left, right, key=None, lkey=None, rkey=None, presorted=False, buffersize=None, tempdir=None, cache=True):
    """
    Return rows from the `left` table where the key value does not occur in
    the `right` table. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [0, 'black'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [4, 'yellow'],
        ...           [5, 'white']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square']]
        >>> table3 = etl.antijoin(table1, table2, key='id')
        >>> table3
        +----+----------+
        | id | colour   |
        +====+==========+
        |  0 | 'black'  |
        +----+----------+
        |  2 | 'red'    |
        +----+----------+
        |  4 | 'yellow' |
        +----+----------+
        |  5 | 'white'  |
        +----+----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    """
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return AntiJoinView(left=left, right=right, lkey=lkey, rkey=rkey, presorted=presorted, buffersize=buffersize, tempdir=tempdir, cache=cache)