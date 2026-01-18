from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_aggregate_empty_key_is_None():
    table = (('foo', 'bar'),)
    aggregators = OrderedDict()
    aggregators['minbar'] = ('bar', min)
    aggregators['maxbar'] = ('bar', max)
    aggregators['sumbar'] = ('bar', sum)
    actual = aggregate(table, None, aggregators)
    expect = (('minbar', 'maxbar', 'sumbar'),)
    ieq(expect, actual)