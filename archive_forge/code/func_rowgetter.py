from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
def rowgetter(*indices):
    if len(indices) == 0:
        return lambda row: tuple()
    elif len(indices) == 1:
        index = indices[0]
        return lambda row: (row[index],)
    else:
        return operator.itemgetter(*indices)