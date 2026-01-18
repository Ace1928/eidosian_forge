from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_crossjoin_prefix():
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'))
    table2 = (('id', 'shape'), (1, 'circle'), (3, 'square'))
    table3 = crossjoin(table1, table2, prefix=True)
    expect3 = (('1_id', '1_colour', '2_id', '2_shape'), (1, 'blue', 1, 'circle'), (1, 'blue', 3, 'square'), (2, 'red', 1, 'circle'), (2, 'red', 3, 'square'))
    ieq(expect3, table3)