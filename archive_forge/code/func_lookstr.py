from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml
def lookstr(table, limit=0, **kwargs):
    """Like :func:`petl.util.vis.look` but use str() rather than repr() for data
    values.

    """
    kwargs['vrepr'] = str
    return look(table, limit=limit, **kwargs)