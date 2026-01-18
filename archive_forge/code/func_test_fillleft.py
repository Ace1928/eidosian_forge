from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl.transform.fills import filldown, fillleft, fillright
def test_fillleft():
    table = (('foo', 'bar', 'baz'), (1, 'a', None), (1, None, 0.23), (1, 'b', None), (2, None, None), (None, None, 0.56), (2, 'c', None), (None, 'c', 0.72))
    actual = fillleft(table)
    expect = (('foo', 'bar', 'baz'), (1, 'a', None), (1, 0.23, 0.23), (1, 'b', None), (2, None, None), (0.56, 0.56, 0.56), (2, 'c', None), ('c', 'c', 0.72))
    ieq(expect, actual)
    ieq(expect, actual)