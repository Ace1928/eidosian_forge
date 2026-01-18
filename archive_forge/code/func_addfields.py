from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def addfields(table, field_defs, missing=None):
    """
    Add fields with fixed or calculated values. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['M', 12],
        ...           ['F', 34],
        ...           ['-', 56]]
        >>> # using a fixed value or a calculation
        ... table2 = etl.addfields(table1,
        ...                        [('baz', 42),
        ...                         ('luhrmann', lambda rec: rec['bar'] * 2)])
        >>> table2
        +-----+-----+-----+----------+
        | foo | bar | baz | luhrmann |
        +=====+=====+=====+==========+
        | 'M' |  12 |  42 |       24 |
        +-----+-----+-----+----------+
        | 'F' |  34 |  42 |       68 |
        +-----+-----+-----+----------+
        | '-' |  56 |  42 |      112 |
        +-----+-----+-----+----------+

        >>> # you can specify an index as a 3rd item in each tuple -- indicies
        ... # are evaluated in order.
        ... table2 = etl.addfields(table1,
        ...                        [('baz', 42, 0),
        ...                         ('luhrmann', lambda rec: rec['bar'] * 2, 0)])
        >>> table2
        +----------+-----+-----+-----+
        | luhrmann | baz | foo | bar |
        +==========+=====+=====+=====+
        |       24 |  42 | 'M' |  12 |
        +----------+-----+-----+-----+
        |       68 |  42 | 'F' |  34 |
        +----------+-----+-----+-----+
        |      112 |  42 | '-' |  56 |
        +----------+-----+-----+-----+

    """
    return AddFieldsView(table, field_defs, missing=missing)