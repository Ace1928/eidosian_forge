from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, iterpeek
from petl.io.numpy import construct_dtype
def frombcolz(source, expression=None, outcols=None, limit=None, skip=0):
    """Extract a table from a bcolz ctable, e.g.::

        >>> import petl as etl
        >>>
        >>> def example_from_bcolz():
        ...     import bcolz
        ...     cols = [
        ...         ['apples', 'oranges', 'pears'],
        ...         [1, 3, 7],
        ...         [2.5, 4.4, .1]
        ...     ]
        ...     names = ('foo', 'bar', 'baz')
        ...     ctbl = bcolz.ctable(cols, names=names)
        ...     return etl.frombcolz(ctbl)
        >>>
        >>> example_from_bcolz() # doctest: +SKIP
        +-----------+-----+-----+
        | foo       | bar | baz |
        +===========+=====+=====+
        | 'apples'  |   1 | 2.5 |
        +-----------+-----+-----+
        | 'oranges' |   3 | 4.4 |
        +-----------+-----+-----+
        | 'pears'   |   7 | 0.1 |
        +-----------+-----+-----+

    If `expression` is provided it will be executed by bcolz and only
    matching rows returned, e.g.::

        >>> tbl2 = etl.frombcolz(ctbl, expression='bar > 1') # doctest: +SKIP
        >>> tbl2 # doctest: +SKIP
        +-----------+-----+-----+
        | foo       | bar | baz |
        +===========+=====+=====+
        | 'oranges' |   3 | 4.4 |
        +-----------+-----+-----+
        | 'pears'   |   7 | 0.1 |
        +-----------+-----+-----+

    .. versionadded:: 1.1.0

    """
    return BcolzView(source, expression=expression, outcols=outcols, limit=limit, skip=skip)