from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
def pushheader(table, header, *args):
    """
    Push rows down and prepend a header row. E.g.::

        >>> import petl as etl
        >>> table1 = [['a', 1],
        ...           ['b', 2]]
        >>> table2 = etl.pushheader(table1, ['foo', 'bar'])
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'b' |   2 |
        +-----+-----+

    The header row can either be a list or positional arguments.

    """
    return PushHeaderView(table, header, *args)