from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from itertools import islice
from petl.compat import izip_longest, text_type, next
from petl.util.base import asindices, Table
def listoflists(tbl):
    return [list(row) for row in tbl]