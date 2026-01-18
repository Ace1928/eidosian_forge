from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_mergeduplicates_compoundkey():
    table = [['foo', 'bar', 'baz'], ['a', 1, True], ['a', 1, True], ['a', 2, False], ['a', 2, None], ['c', 3, True], ['c', 3, False]]
    actual = mergeduplicates(table, key=('foo', 'bar'))
    expect = [('foo', 'bar', 'baz'), ('a', 1, True), ('a', 2, False), ('c', 3, Conflict([True, False]))]
    ieq(expect, actual)