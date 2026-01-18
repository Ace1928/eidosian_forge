from __future__ import absolute_import, print_function, division
from petl.compat import next
from petl.util.base import Table, asindices
def iterfilldown(table, fillfields, missing):
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        return
    yield tuple(hdr)
    if not fillfields:
        fillfields = hdr
    fillindices = asindices(hdr, fillfields)
    fill = list(next(it))
    yield tuple(fill)
    for row in it:
        outrow = list(row)
        for idx in fillindices:
            if row[idx] == missing:
                outrow[idx] = fill[idx]
            else:
                fill[idx] = row[idx]
        yield tuple(outrow)