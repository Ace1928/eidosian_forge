from __future__ import absolute_import, print_function, division
import itertools
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.comparison import comparable_itemgetter, Comparable
from petl.util.base import Table, asindices, rowgetter, rowgroupby, \
from petl.transform.sorts import sort
from petl.transform.basics import cut, cutout
from petl.transform.dedup import distinct
def itercrossjoin(sources, prefix):
    outhdr = list()
    for i, s in enumerate(sources):
        if prefix:
            outhdr.extend([text_type(i + 1) + '_' + text_type(f) for f in header(s)])
        else:
            outhdr.extend(header(s))
    yield tuple(outhdr)
    datasrcs = [data(src) for src in sources]
    for prod in itertools.product(*datasrcs):
        outrow = list()
        for row in prod:
            outrow.extend(row)
        yield tuple(outrow)