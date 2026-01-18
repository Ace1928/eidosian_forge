from __future__ import absolute_import, print_function, division
from petl.util.base import values, header, Table
def diffheaders(t1, t2):
    """
    Return the difference between the headers of the two tables as a pair of
    sets. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['a', 1, .3]]
        >>> table2 = [['baz', 'bar', 'quux'],
        ...           ['a', 1, .3]]
        >>> add, sub = etl.diffheaders(table1, table2)
        >>> add
        {'quux'}
        >>> sub
        {'foo'}

    """
    t1h = set(header(t1))
    t2h = set(header(t2))
    return (t2h - t1h, t1h - t2h)