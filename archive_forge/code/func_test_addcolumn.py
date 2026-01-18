from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addcolumn():
    table1 = (('foo', 'bar'), ('A', 1), ('B', 2))
    col = [True, False]
    expect2 = (('foo', 'bar', 'baz'), ('A', 1, True), ('B', 2, False))
    table2 = addcolumn(table1, 'baz', col)
    ieq(expect2, table2)
    ieq(expect2, table2)
    table3 = (('foo', 'bar'), ('A', 1), ('B', 2), ('C', 2))
    expect4 = (('foo', 'bar', 'baz'), ('A', 1, True), ('B', 2, False), ('C', 2, None))
    table4 = addcolumn(table3, 'baz', col)
    ieq(expect4, table4)
    col = [True, False, False]
    expect5 = (('foo', 'bar', 'baz'), ('A', 1, True), ('B', 2, False), (None, None, False))
    table5 = addcolumn(table1, 'baz', col)
    ieq(expect5, table5)