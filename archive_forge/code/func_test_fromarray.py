from __future__ import absolute_import, print_function, division
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_, assert_almost_equal
from petl.io.numpy import toarray, fromarray, torecarray
def test_fromarray():
    t = [('foo', 'bar', 'baz'), ('apples', 1, 2.5), ('oranges', 3, 4.4), ('pears', 7, 0.1)]
    a = toarray(t)
    u = fromarray(a)
    ieq(t, u)