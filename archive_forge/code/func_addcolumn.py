from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def addcolumn(table, field, col, index=None, missing=None):
    """
    Add a column of data to the table. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['A', 1],
        ...           ['B', 2]]
        >>> col = [True, False]
        >>> table2 = etl.addcolumn(table1, 'baz', col)
        >>> table2
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'B' |   2 | False |
        +-----+-----+-------+

    Use the `index` parameter to control the position of the new column.

    """
    return AddColumnView(table, field, col, index=index, missing=missing)