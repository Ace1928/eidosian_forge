from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, itervalues
from petl.transform.sorts import sort
def iterunique(source, key):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    yield tuple(hdr)
    if key is None:
        indices = range(len(hdr))
    else:
        indices = asindices(hdr, key)
    getkey = operator.itemgetter(*indices)
    try:
        prev = next(it)
    except StopIteration:
        return
    prev_key = getkey(prev)
    prev_comp_ne = True
    for curr in it:
        curr_key = getkey(curr)
        curr_comp_ne = curr_key != prev_key
        if prev_comp_ne and curr_comp_ne:
            yield tuple(prev)
        prev = curr
        prev_key = curr_key
        prev_comp_ne = curr_comp_ne
    if prev_comp_ne:
        yield prev