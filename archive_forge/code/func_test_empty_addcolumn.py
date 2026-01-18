from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_empty_addcolumn():
    table1 = empty()
    table2 = addcolumn(table1, 'foo', ['A', 'B'])
    table3 = addcolumn(table2, 'bar', [1, 2])
    expect = (('foo', 'bar'), ('A', 1), ('B', 2))
    ieq(expect, table3)
    ieq(expect, table3)