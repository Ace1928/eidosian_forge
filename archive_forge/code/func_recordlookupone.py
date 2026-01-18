from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.errors import DuplicateKeyError
from petl.util.base import Table, asindices, asdict, Record, rowgetter
def recordlookupone(table, key, dictionary=None, strict=False):
    """
    Load a dictionary with data from the given table, mapping to record objects,
    assuming there is at most one row for each key.

    """
    if dictionary is None:
        dictionary = dict()
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    keyindices = asindices(hdr, key)
    assert len(keyindices) > 0, 'no key selected'
    getkey = operator.itemgetter(*keyindices)
    for row in it:
        k = getkey(row)
        if strict and k in dictionary:
            raise DuplicateKeyError(k)
        elif k not in dictionary:
            d = Record(row, flds)
            dictionary[k] = d
    return dictionary