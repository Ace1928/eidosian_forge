from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def rowlengths(table):
    """
    Report on row lengths found in the table. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar', 'baz'],
        ...          ['A', 1, 2],
        ...          ['B', '2', '3.4'],
        ...          [u'B', u'3', u'7.8', True],
        ...          ['D', 'xyz', 9.0],
        ...          ['E', None],
        ...          ['F', 9]]
        >>> etl.rowlengths(table)
        +--------+-------+
        | length | count |
        +========+=======+
        |      3 |     3 |
        +--------+-------+
        |      2 |     2 |
        +--------+-------+
        |      4 |     1 |
        +--------+-------+

    Useful for finding potential problems in data files.

    """
    counter = Counter()
    for row in data(table):
        counter[len(row)] += 1
    output = [('length', 'count')]
    output.extend(counter.most_common())
    return wrap(output)