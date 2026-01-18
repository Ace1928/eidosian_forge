from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def facetrecordtrees(table, key, start='start', stop='stop'):
    """
    Construct faceted interval trees for the given table, where each node in 
    the tree is a record.

    """
    import intervaltree
    getstart = attrgetter(start)
    getstop = attrgetter(stop)
    getkey = attrgetter(key)
    trees = dict()
    for rec in records(table):
        k = getkey(rec)
        if k not in trees:
            trees[k] = intervaltree.IntervalTree()
        trees[k].addi(getstart(rec), getstop(rec), rec)
    return trees