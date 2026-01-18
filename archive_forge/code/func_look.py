from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml
def look(table, limit=0, vrepr=None, index_header=None, style=None, truncate=None, width=None):
    """
    Format a portion of the table as text for inspection in an interactive
    session. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2]]
        >>> etl.look(table1)
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' |   1 |
        +-----+-----+
        | 'b' |   2 |
        +-----+-----+

        >>> # alternative formatting styles
        ... etl.look(table1, style='simple')
        ===  ===
        foo  bar
        ===  ===
        'a'    1
        'b'    2
        ===  ===

        >>> etl.look(table1, style='minimal')
        foo  bar
        'a'    1
        'b'    2

        >>> # any irregularities in the length of header and/or data
        ... # rows will appear as blank cells
        ... table2 = [['foo', 'bar'],
        ...           ['a'],
        ...           ['b', 2, True]]
        >>> etl.look(table2)
        +-----+-----+------+
        | foo | bar |      |
        +=====+=====+======+
        | 'a' |     |      |
        +-----+-----+------+
        | 'b' |   2 | True |
        +-----+-----+------+

    Three alternative presentation styles are available: 'grid', 'simple' and
    'minimal', where 'grid' is the default. A different style can be specified
    using the `style` keyword argument. The default style can also be changed
    by setting ``petl.config.look_style``.

    """
    if limit == 0:
        limit = config.look_limit
    if vrepr is None:
        vrepr = config.look_vrepr
    if index_header is None:
        index_header = config.look_index_header
    if style is None:
        style = config.look_style
    if width is None:
        width = config.look_width
    return Look(table, limit=limit, vrepr=vrepr, index_header=index_header, style=style, truncate=truncate, width=width)