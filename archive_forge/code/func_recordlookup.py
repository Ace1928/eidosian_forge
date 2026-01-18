from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.errors import DuplicateKeyError
from petl.util.base import Table, asindices, asdict, Record, rowgetter
def recordlookup(table, key, dictionary=None):
    """
    Load a dictionary with data from the given table, mapping to record objects.

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
        rec = Record(row, flds)
        if k in dictionary:
            l = dictionary[k]
            l.append(rec)
            dictionary[k] = l
        else:
            dictionary[k] = [rec]
    return dictionary