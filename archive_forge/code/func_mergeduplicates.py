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
def mergeduplicates(table, key, missing=None, presorted=False, buffersize=None, tempdir=None, cache=True):
    """
    Merge duplicate rows under the given key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['A', 1, 2.7],
        ...           ['B', 2, None],
        ...           ['D', 3, 9.4],
        ...           ['B', None, 7.8],
        ...           ['E', None, 42.],
        ...           ['D', 3, 12.3],
        ...           ['A', 2, None]]
        >>> table2 = etl.mergeduplicates(table1, 'foo')
        >>> table2
        +-----+------------------+-----------------------+
        | foo | bar              | baz                   |
        +=====+==================+=======================+
        | 'A' | Conflict({1, 2}) |                   2.7 |
        +-----+------------------+-----------------------+
        | 'B' |                2 |                   7.8 |
        +-----+------------------+-----------------------+
        | 'D' |                3 | Conflict({9.4, 12.3}) |
        +-----+------------------+-----------------------+
        | 'E' | None             |                  42.0 |
        +-----+------------------+-----------------------+

    Missing values are overridden by non-missing values. Conflicting values are
    reported as an instance of the Conflict class (sub-class of frozenset).

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    See also :func:`petl.transform.dedup.conflicts`.

    """
    return MergeDuplicatesView(table, key, missing=missing, presorted=presorted, buffersize=buffersize, tempdir=tempdir, cache=cache)