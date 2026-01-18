from __future__ import absolute_import, print_function, division
import logging
from petl.test.helpers import eq_
import petl as etl
from petl.util.vis import look, see, lookstr
def test_look_style_simple():
    table = (('foo', 'bar'), ('a', 1), ('b', 2))
    actual = repr(look(table, style='simple'))
    expect = "===  ===\nfoo  bar\n===  ===\n'a'    1\n'b'    2\n===  ===\n"
    eq_(expect, actual)
    etl.config.look_style = 'simple'
    actual = repr(look(table))
    eq_(expect, actual)
    etl.config.look_style = 'grid'