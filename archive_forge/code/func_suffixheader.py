from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
def suffixheader(table, suffix):
    """Suffix all fields in the table header."""
    return SuffixHeaderView(table, suffix)