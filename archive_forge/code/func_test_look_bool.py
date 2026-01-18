from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_look_bool():
    table = (('foo', 'bar'), ('a', True), ('b', False))
    actual = repr(look(table))
    expect = "+-----+-------+\n| foo | bar   |\n+=====+=======+\n| 'a' | True  |\n+-----+-------+\n| 'b' | False |\n+-----+-------+\n"
    eq_(expect, actual)