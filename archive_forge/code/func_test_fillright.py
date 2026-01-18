from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl.transform.fills import filldown, fillleft, fillright
def test_fillright():
    table = (('foo', 'bar', 'baz'), (1, 'a', None), (1, None, 0.23), (1, 'b', None), (2, None, None), (2, None, 0.56), (2, 'c', None), (None, 'c', 0.72))
    actual = fillright(table)
    expect = (('foo', 'bar', 'baz'), (1, 'a', 'a'), (1, 1, 0.23), (1, 'b', 'b'), (2, 2, 2), (2, 2, 0.56), (2, 'c', 'c'), (None, 'c', 0.72))
    ieq(expect, actual)
    ieq(expect, actual)