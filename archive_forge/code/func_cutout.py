from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def cutout(table, *args, **kwargs):
    """
    Remove fields. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['A', 1, 2.7],
        ...           ['B', 2, 3.4],
        ...           ['B', 3, 7.8],
        ...           ['D', 42, 9.0],
        ...           ['E', 12]]
        >>> table2 = etl.cutout(table1, 'bar')
        >>> table2
        +-----+------+
        | foo | baz  |
        +=====+======+
        | 'A' |  2.7 |
        +-----+------+
        | 'B' |  3.4 |
        +-----+------+
        | 'B' |  7.8 |
        +-----+------+
        | 'D' |  9.0 |
        +-----+------+
        | 'E' | None |
        +-----+------+

    See also :func:`petl.transform.basics.cut`.

    """
    return CutOutView(table, args, **kwargs)