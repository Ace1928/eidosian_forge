from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_aggregate_multiple_source_fields_key_is_None():
    table = (('foo', 'bar', 'baz'), ('a', 3, True), ('a', 7, False), ('b', 2, True), ('b', 2, False), ('b', 9, False), ('c', 4, True))
    expect = (('value',), ([(3, True), (7, False), (2, True), (2, False), (9, False), (4, True)],))
    actual = aggregate(table, None, list, ('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)
    actual = aggregate(table, key=None, aggregation=list, value=('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)
    actual = aggregate(table, key=None)
    actual['value'] = (('bar', 'baz'), list)
    ieq(expect, actual)
    ieq(expect, actual)