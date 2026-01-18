from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def annex(*tables, **kwargs):
    """
    Join two or more tables by row order. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['A', 9],
        ...           ['C', 2],
        ...           ['F', 1]]
        >>> table2 = [['foo', 'baz'],
        ...           ['B', 3],
        ...           ['D', 10]]
        >>> table3 = etl.annex(table1, table2)
        >>> table3
        +-----+-----+------+------+
        | foo | bar | foo  | baz  |
        +=====+=====+======+======+
        | 'A' |   9 | 'B'  |    3 |
        +-----+-----+------+------+
        | 'C' |   2 | 'D'  |   10 |
        +-----+-----+------+------+
        | 'F' |   1 | None | None |
        +-----+-----+------+------+

    See also :func:`petl.transform.joins.join`.

    """
    return AnnexView(tables, **kwargs)