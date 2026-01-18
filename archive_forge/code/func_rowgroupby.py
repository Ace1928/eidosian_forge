from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
def rowgroupby(table, key, value=None):
    """Convenient adapter for :func:`itertools.groupby`. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['a', 1, True],
        ...           ['b', 3, True],
        ...           ['b', 2]]
        >>> # group entire rows
        ... for key, group in etl.rowgroupby(table1, 'foo'):
        ...     print(key, list(group))
        ...
        a [('a', 1, True)]
        b [('b', 3, True), ('b', 2)]
        >>> # group specific values
        ... for key, group in etl.rowgroupby(table1, 'foo', 'bar'):
        ...     print(key, list(group))
        ...
        a [1]
        b [3, 2]

    N.B., assumes the input table is already sorted by the given key.

    """
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    it = (Record(row, flds) for row in it)
    if callable(key):
        getkey = key
        native_key = True
    else:
        kindices = asindices(hdr, key)
        getkey = comparable_itemgetter(*kindices)
        native_key = False
    git = groupby(it, key=getkey)
    if value is None:
        if native_key:
            return git
        else:
            return ((k.inner, vals) for k, vals in git)
    else:
        if callable(value):
            getval = value
        else:
            vindices = asindices(hdr, value)
            getval = operator.itemgetter(*vindices)
        if native_key:
            return ((k, (getval(v) for v in vals)) for k, vals in git)
        else:
            return ((k.inner, (getval(v) for v in vals)) for k, vals in git)