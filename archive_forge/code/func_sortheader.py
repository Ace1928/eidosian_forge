from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
def sortheader(table, reverse=False, missing=None):
    """Re-order columns so the header is sorted.

    .. versionadded:: 1.1.0

    """
    return SortHeaderView(table, reverse, missing)