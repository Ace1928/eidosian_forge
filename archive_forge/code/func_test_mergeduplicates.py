from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_mergeduplicates():
    table = (('foo', 'bar', 'baz'), ('A', 1, 2), ('B', '2', None), ('D', 'xyz', 9.4), ('B', None, u'7.8', True), ('E', None, 42.0), ('D', 'xyz', 12.3), ('A', 2, None))
    result = mergeduplicates(table, 'foo', missing=None)
    expectation = (('foo', 'bar', 'baz'), ('A', Conflict([1, 2]), 2), ('B', '2', u'7.8'), ('D', 'xyz', Conflict([9.4, 12.3])), ('E', None, 42.0))
    ieq(expectation, result)