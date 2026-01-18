from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_look_truncate():
    table = (('foo', 'bar'), ('abcd', 1234), ('bcde', 2345))
    actual = repr(look(table, truncate=3))
    expect = "+-----+-----+\n| foo | bar |\n+=====+=====+\n| 'ab | 123 |\n+-----+-----+\n| 'bc | 234 |\n+-----+-----+\n"
    eq_(expect, actual)
    actual = repr(look(table, truncate=3, vrepr=str))
    expect = '+-----+-----+\n| foo | bar |\n+=====+=====+\n| abc | 123 |\n+-----+-----+\n| bcd | 234 |\n+-----+-----+\n'
    eq_(expect, actual)