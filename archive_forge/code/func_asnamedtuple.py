from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
def asnamedtuple(nt, row, missing=None):
    try:
        return nt(*row)
    except TypeError:
        ne = len(nt._fields)
        na = len(row)
        if ne > na:
            padded = tuple(row) + (missing,) * (ne - na)
            return nt(*padded)
        elif ne < na:
            return nt(*row[:ne])
        else:
            raise