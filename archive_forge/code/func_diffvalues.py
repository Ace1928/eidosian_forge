from __future__ import absolute_import, print_function, division
from petl.util.base import values, header, Table
def diffvalues(t1, t2, f):
    """
    Return the difference between the values under the given field in the two
    tables, e.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 3]]
        >>> table2 = [['bar', 'foo'],
        ...           [1, 'a'],
        ...           [3, 'c']]
        >>> add, sub = etl.diffvalues(table1, table2, 'foo')
        >>> add
        {'c'}
        >>> sub
        {'b'}

    """
    t1v = set(values(t1, f))
    t2v = set(values(t2, f))
    return (t2v - t1v, t1v - t2v)