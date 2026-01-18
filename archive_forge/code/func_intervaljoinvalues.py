from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def intervaljoinvalues(left, right, value, lstart='start', lstop='stop', rstart='start', rstop='stop', lkey=None, rkey=None, include_stop=False):
    """
    Convenience function to join the left table with values from a specific 
    field in the right hand table.
    
    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """
    assert (lkey is None) == (rkey is None), 'facet key field must be provided for both or neither table'
    if lkey is None:
        lkp = intervallookup(right, start=rstart, stop=rstop, value=value, include_stop=include_stop)
        f = lambda row: lkp.search(row[lstart], row[lstop])
    else:
        lkp = facetintervallookup(right, rkey, start=rstart, stop=rstop, value=value, include_stop=include_stop)
        f = lambda row: lkp[row[lkey]].search(row[lstart], row[lstop])
    return addfield(left, value, f)