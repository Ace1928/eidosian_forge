from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def skipcomments(table, prefix):
    """
    Skip any row where the first value is a string and starts with
    `prefix`. E.g.::

        >>> import petl as etl
        >>> table1 = [['##aaa', 'bbb', 'ccc'],
        ...           ['##mmm',],
        ...           ['#foo', 'bar'],
        ...           ['##nnn', 1],
        ...           ['a', 1],
        ...           ['b', 2]]
        >>> table2 = etl.skipcomments(table1, '##')
        >>> table2
        +------+-----+
        | #foo | bar |
        +======+=====+
        | 'a'  |   1 |
        +------+-----+
        | 'b'  |   2 |
        +------+-----+

    Use the `prefix` parameter to determine which string to consider as
    indicating a comment.

    """
    return SkipCommentsView(table, prefix)