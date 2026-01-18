from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
def test_aggregate_multifield():
    table1 = (('foo', 'bar'), ('a', 3), ('a', 7), ('b', 2), ('b', 1), ('b', 9), ('c', 4))
    aggregators = OrderedDict()
    aggregators['count'] = len
    aggregators['minbar'] = ('bar', min)
    aggregators['maxbar'] = ('bar', max)
    aggregators['sumbar'] = ('bar', sum)
    aggregators['listbar'] = ('bar', list)
    aggregators['bars'] = ('bar', strjoin(', '))
    table2 = aggregate(table1, 'foo', aggregators)
    expect2 = (('foo', 'count', 'minbar', 'maxbar', 'sumbar', 'listbar', 'bars'), ('a', 2, 3, 7, 10, [3, 7], '3, 7'), ('b', 3, 1, 9, 12, [2, 1, 9], '2, 1, 9'), ('c', 1, 4, 4, 4, [4], '4'))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table3 = aggregate(table1, 'foo')
    table3['count'] = len
    table3['minbar'] = ('bar', min)
    table3['maxbar'] = ('bar', max)
    table3['sumbar'] = ('bar', sum)
    table3['listbar'] = 'bar'
    table3['bars'] = ('bar', strjoin(', '))
    ieq(expect2, table3)
    aggregators = [('count', len), ('minbar', 'bar', min), ('maxbar', 'bar', max), ('sumbar', 'bar', sum), ('listbar', 'bar', list), ('bars', 'bar', strjoin(', '))]
    table4 = aggregate(table1, 'foo', aggregators)
    ieq(expect2, table4)
    ieq(expect2, table4)