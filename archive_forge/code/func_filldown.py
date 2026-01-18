from __future__ import absolute_import, print_function, division
from petl.compat import next
from petl.util.base import Table, asindices
def filldown(table, *fields, **kwargs):
    """
    Replace missing values with non-missing values from the row above. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           [1, 'a', None],
        ...           [1, None, .23],
        ...           [1, 'b', None],
        ...           [2, None, None],
        ...           [2, None, .56],
        ...           [2, 'c', None],
        ...           [None, 'c', .72]]
        >>> table2 = etl.filldown(table1)
        >>> table2.lookall()
        +-----+-----+------+
        | foo | bar | baz  |
        +=====+=====+======+
        |   1 | 'a' | None |
        +-----+-----+------+
        |   1 | 'a' | 0.23 |
        +-----+-----+------+
        |   1 | 'b' | 0.23 |
        +-----+-----+------+
        |   2 | 'b' | 0.23 |
        +-----+-----+------+
        |   2 | 'b' | 0.56 |
        +-----+-----+------+
        |   2 | 'c' | 0.56 |
        +-----+-----+------+
        |   2 | 'c' | 0.72 |
        +-----+-----+------+

        >>> table3 = etl.filldown(table1, 'bar')
        >>> table3.lookall()
        +------+-----+------+
        | foo  | bar | baz  |
        +======+=====+======+
        |    1 | 'a' | None |
        +------+-----+------+
        |    1 | 'a' | 0.23 |
        +------+-----+------+
        |    1 | 'b' | None |
        +------+-----+------+
        |    2 | 'b' | None |
        +------+-----+------+
        |    2 | 'b' | 0.56 |
        +------+-----+------+
        |    2 | 'c' | None |
        +------+-----+------+
        | None | 'c' | 0.72 |
        +------+-----+------+

        >>> table4 = etl.filldown(table1, 'bar', 'baz')
        >>> table4.lookall()
        +------+-----+------+
        | foo  | bar | baz  |
        +======+=====+======+
        |    1 | 'a' | None |
        +------+-----+------+
        |    1 | 'a' | 0.23 |
        +------+-----+------+
        |    1 | 'b' | 0.23 |
        +------+-----+------+
        |    2 | 'b' | 0.23 |
        +------+-----+------+
        |    2 | 'b' | 0.56 |
        +------+-----+------+
        |    2 | 'c' | 0.56 |
        +------+-----+------+
        | None | 'c' | 0.72 |
        +------+-----+------+

    Use the `missing` keyword argument to control which value is treated as
    missing (`None` by default).

    """
    return FillDownView(table, fields, **kwargs)