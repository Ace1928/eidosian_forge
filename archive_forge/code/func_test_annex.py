from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_annex():
    table1 = (('foo', 'bar'), ('A', 9), ('C', 2), ('F', 1))
    table2 = (('foo', 'baz'), ('B', 3), ('D', 10))
    expect = (('foo', 'bar', 'foo', 'baz'), ('A', 9, 'B', 3), ('C', 2, 'D', 10), ('F', 1, None, None))
    actual = annex(table1, table2)
    ieq(expect, actual)
    ieq(expect, actual)
    expect21 = (('foo', 'baz', 'foo', 'bar'), ('B', 3, 'A', 9), ('D', 10, 'C', 2), (None, None, 'F', 1))
    actual21 = annex(table2, table1)
    ieq(expect21, actual21)
    ieq(expect21, actual21)