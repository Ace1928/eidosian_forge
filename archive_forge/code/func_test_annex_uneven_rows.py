from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_annex_uneven_rows():
    table1 = (('foo', 'bar'), ('A', 9, True), ('C', 2), ('F',))
    table2 = (('foo', 'baz'), ('B', 3), ('D', 10))
    expect = (('foo', 'bar', 'foo', 'baz'), ('A', 9, 'B', 3), ('C', 2, 'D', 10), ('F', None, None, None))
    actual = annex(table1, table2)
    ieq(expect, actual)
    ieq(expect, actual)