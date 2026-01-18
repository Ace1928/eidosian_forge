from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl.transform.fills import filldown, fillleft, fillright
def test_filldown():
    table = (('foo', 'bar', 'baz'), (1, 'a', None), (1, None, 0.23), (1, 'b', None), (2, None, None), (2, None, 0.56), (2, 'c', None), (None, 'c', 0.72))
    actual = filldown(table)
    expect = (('foo', 'bar', 'baz'), (1, 'a', None), (1, 'a', 0.23), (1, 'b', 0.23), (2, 'b', 0.23), (2, 'b', 0.56), (2, 'c', 0.56), (2, 'c', 0.72))
    ieq(expect, actual)
    ieq(expect, actual)
    actual = filldown(table, 'bar')
    expect = (('foo', 'bar', 'baz'), (1, 'a', None), (1, 'a', 0.23), (1, 'b', None), (2, 'b', None), (2, 'b', 0.56), (2, 'c', None), (None, 'c', 0.72))
    ieq(expect, actual)
    ieq(expect, actual)
    actual = filldown(table, 'foo', 'bar')
    expect = (('foo', 'bar', 'baz'), (1, 'a', None), (1, 'a', 0.23), (1, 'b', None), (2, 'b', None), (2, 'b', 0.56), (2, 'c', None), (2, 'c', 0.72))
    ieq(expect, actual)
    ieq(expect, actual)