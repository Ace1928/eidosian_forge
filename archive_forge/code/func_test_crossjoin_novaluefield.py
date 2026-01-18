from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_crossjoin_novaluefield():
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'))
    expect = (('id', 'colour', 'id', 'shape'), (1, 'blue', 1, 'circle'), (1, 'blue', 3, 'square'), (2, 'red', 1, 'circle'), (2, 'red', 3, 'square'))
    actual = crossjoin(table1, table2, key='id')
    ieq(expect, actual)
    actual = crossjoin(cut(table1, 'id'), table2, key='id')
    ieq(cut(expect, 0, 2, 'shape'), actual)
    actual = crossjoin(table1, cut(table2, 'id'), key='id')
    ieq(cut(expect, 0, 'colour', 2), actual)
    actual = crossjoin(cut(table1, 'id'), cut(table2, 'id'), key='id')
    ieq(cut(expect, 0, 2), actual)