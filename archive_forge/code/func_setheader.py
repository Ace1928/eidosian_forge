from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
def setheader(table, header):
    """
    Replace header row in the given table. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2]]
        >>> table2 = etl.setheader(table1, ['foofoo', 'barbar'])
        >>> table2
        +--------+--------+
        | foofoo | barbar |
        +========+========+
        | 'a'    |      1 |
        +--------+--------+
        | 'b'    |      2 |
        +--------+--------+

    See also :func:`petl.transform.headers.extendheader`,
    :func:`petl.transform.headers.pushheader`.

    """
    return SetHeaderView(table, header)