from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_crossjoin():
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'))
    table3 = crossjoin(table1, table2)
    expect3 = (('id', 'colour', 'id', 'shape'), (1, 'blue', 1, 'circle'), (1, 'blue', 3, 'square'), (2, 'red', 1, 'circle'), (2, 'red', 3, 'square'))
    ieq(expect3, table3)