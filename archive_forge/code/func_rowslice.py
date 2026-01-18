from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def rowslice(table, *sliceargs):
    """
    Choose a subsequence of data rows. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['c', 5],
        ...           ['d', 7],
        ...           ['f', 42]]
        >>> table2 = etl.rowslice(table1, 2)
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'b' |   2 |
        +-----+-----+

        >>> table3 = etl.rowslice(table1, 1, 4)
        >>> table3
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'b' |   2 |
        +-----+-----+
        | 'c' |   5 |
        +-----+-----+
        | 'd' |   7 |
        +-----+-----+

        >>> table4 = etl.rowslice(table1, 0, 5, 2)
        >>> table4
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'c' |   5 |
        +-----+-----+
        | 'f' |  42 |
        +-----+-----+

    Positional arguments are used to slice the data rows. The `sliceargs` are
    passed through to :func:`itertools.islice`.

    See also :func:`petl.transform.basics.head`,
    :func:`petl.transform.basics.tail`.

    """
    return RowSliceView(table, *sliceargs)