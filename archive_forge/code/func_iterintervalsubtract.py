from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def iterintervalsubtract(left, right, lstart, lstop, rstart, rstop, lkey, rkey, include_stop):
    lit = iter(left)
    lhdr = next(lit)
    lflds = list(map(text_type, lhdr))
    rit = iter(right)
    rhdr = next(rit)
    asindices(lhdr, lstart)
    asindices(lhdr, lstop)
    if lkey is not None:
        asindices(lhdr, lkey)
    asindices(rhdr, rstart)
    asindices(rhdr, rstop)
    if rkey is not None:
        asindices(rhdr, rkey)
    outhdr = list(lflds)
    yield tuple(outhdr)
    lstartidx, lstopidx = asindices(lhdr, (lstart, lstop))
    getlcoords = itemgetter(lstartidx, lstopidx)
    getrcoords = itemgetter(*asindices(rhdr, (rstart, rstop)))
    if rkey is None:
        lookup = intervallookup(right, rstart, rstop, include_stop=include_stop)
        search = lookup.search
        for lrow in lit:
            start, stop = getlcoords(lrow)
            rrows = search(start, stop)
            if not rrows:
                yield tuple(lrow)
            else:
                rivs = sorted([getrcoords(rrow) for rrow in rrows], key=itemgetter(0))
                for x, y in _subtract(start, stop, rivs):
                    out = list(lrow)
                    out[lstartidx] = x
                    out[lstopidx] = y
                    yield tuple(out)
    else:
        lookup = facetintervallookup(right, key=rkey, start=rstart, stop=rstop, include_stop=include_stop)
        getlkey = itemgetter(*asindices(lhdr, lkey))
        for lrow in lit:
            lkey = getlkey(lrow)
            start, stop = getlcoords(lrow)
            try:
                rrows = lookup[lkey].search(start, stop)
            except KeyError:
                rrows = None
            except AttributeError:
                rrows = None
            if not rrows:
                yield tuple(lrow)
            else:
                rivs = sorted([getrcoords(rrow) for rrow in rrows], key=itemgetter(0))
                for x, y in _subtract(start, stop, rivs):
                    out = list(lrow)
                    out[lstartidx] = x
                    out[lstopidx] = y
                    yield tuple(out)