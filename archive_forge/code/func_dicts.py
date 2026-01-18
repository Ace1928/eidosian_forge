from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
def dicts(table, *sliceargs, **kwargs):
    """
    Return a container supporting iteration over rows as dicts. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar'], ['a', 1], ['b', 2]]
        >>> d = etl.dicts(table)
        >>> d
        {'foo': 'a', 'bar': 1}
        {'foo': 'b', 'bar': 2}
        >>> list(d)
        [{'foo': 'a', 'bar': 1}, {'foo': 'b', 'bar': 2}]

    Short rows are padded with the value of the `missing` keyword argument.

    """
    return DictsView(table, *sliceargs, **kwargs)