from __future__ import division, print_function, absolute_import
import locale
import codecs
from petl.compat import izip_longest
from petl.util.base import Table
def fromcolumns(cols, header=None, missing=None):
    """View a sequence of columns as a table, e.g.::

        >>> import petl as etl
        >>> cols = [[0, 1, 2], ['a', 'b', 'c']]
        >>> tbl = etl.fromcolumns(cols)
        >>> tbl
        +----+-----+
        | f0 | f1  |
        +====+=====+
        |  0 | 'a' |
        +----+-----+
        |  1 | 'b' |
        +----+-----+
        |  2 | 'c' |
        +----+-----+

    If columns are not the same length, values will be padded to the length
    of the longest column with `missing`, which is None by default, e.g.::

        >>> cols = [[0, 1, 2], ['a', 'b']]
        >>> tbl = etl.fromcolumns(cols, missing='NA')
        >>> tbl
        +----+------+
        | f0 | f1   |
        +====+======+
        |  0 | 'a'  |
        +----+------+
        |  1 | 'b'  |
        +----+------+
        |  2 | 'NA' |
        +----+------+

    See also :func:`petl.io.json.fromdicts`.

    .. versionadded:: 1.1.0

    """
    return ColumnsView(cols, header=header, missing=missing)