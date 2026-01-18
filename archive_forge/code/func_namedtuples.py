from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
def namedtuples(table, *sliceargs, **kwargs):
    """
    View the table as a container of named tuples. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar'], ['a', 1], ['b', 2]]
        >>> d = etl.namedtuples(table)
        >>> d
        row(foo='a', bar=1)
        row(foo='b', bar=2)
        >>> list(d)
        [row(foo='a', bar=1), row(foo='b', bar=2)]

    Short rows are padded with the value of the `missing` keyword argument.

    The `name` keyword argument can be given to override the name of the
    named tuple class (defaults to 'row').

    """
    return NamedTuplesView(table, *sliceargs, **kwargs)