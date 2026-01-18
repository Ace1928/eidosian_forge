from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_cat_dupfields():
    table1 = (('foo', 'foo'), (1, 'A'), (2,), (3, 'B', True))
    actual = cat(table1)
    expect = (('foo', 'foo'), (1, 1), (2, 2), (3, 3))
    ieq(expect, actual)
    table2 = (('foo', 'foo', 'bar'), (4, 'C', True), (5, 'D', False))
    actual = cat(table1, table2)
    expect = (('foo', 'foo', 'bar'), (1, 1, None), (2, 2, None), (3, 3, None), (4, 4, True), (5, 5, False))
    ieq(expect, actual)