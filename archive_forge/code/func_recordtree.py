from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def recordtree(table, start='start', stop='stop'):
    """
    Construct an interval tree for the given table, where each node in the
    tree is a row of the table represented as a record object.

    """
    import intervaltree
    getstart = attrgetter(start)
    getstop = attrgetter(stop)
    tree = intervaltree.IntervalTree()
    for rec in records(table):
        tree.addi(getstart(rec), getstop(rec), rec)
    return tree