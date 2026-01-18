from __future__ import absolute_import, print_function, division
from petl.test.helpers import eq_
from petl.compat import PY2
from petl.util.misc import typeset, diffvalues, diffheaders
def test_diffvalues():
    table1 = (('foo', 'bar'), ('a', 1), ('b', 3))
    table2 = (('bar', 'foo'), (1, 'a'), (3, 'c'))
    add, sub = diffvalues(table1, table2, 'foo')
    eq_({'c'}, add)
    eq_({'b'}, sub)