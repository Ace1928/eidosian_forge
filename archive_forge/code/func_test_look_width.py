from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_look_width():
    table = (('foo', 'bar'), ('a', 1), ('b', 2))
    actual = repr(look(table, width=10))
    expect = "+-----+---\n| foo | ba\n+=====+===\n| 'a' |   \n+-----+---\n| 'b' |   \n+-----+---\n"
    eq_(expect, actual)