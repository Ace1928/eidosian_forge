from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
def asindices(hdr, spec):
    """Convert the given field `spec` into a list of field indices."""
    flds = list(map(text_type, hdr))
    indices = list()
    if not isinstance(spec, (list, tuple)):
        spec = (spec,)
    for s in spec:
        if isinstance(s, int) and s < len(hdr):
            indices.append(s)
        elif s in flds:
            idx = flds.index(s)
            indices.append(idx)
            flds[idx] = None
        else:
            raise FieldSelectionError(s)
    return indices