from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl.transform.fills import filldown, fillleft, fillright
def test_filldown_headerless():
    table = []
    actual = filldown(table, 'foo')
    expect = []
    ieq(expect, actual)