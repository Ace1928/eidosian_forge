from __future__ import absolute_import, print_function, division
from petl.test.helpers import eq_
from petl.compat import PY2
from petl.util.misc import typeset, diffvalues, diffheaders
def test_diffheaders():
    table1 = (('foo', 'bar', 'baz'), ('a', 1, 0.3))
    table2 = (('baz', 'bar', 'quux'), ('a', 1, 0.3))
    add, sub = diffheaders(table1, table2)
    eq_({'quux'}, add)
    eq_({'foo'}, sub)