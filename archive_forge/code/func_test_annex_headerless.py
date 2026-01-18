from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_annex_headerless():
    table1 = (('foo', 'bar'), ('C', 2))
    table2 = ()
    expect = table1
    actual = annex(table1, table2)
    ieq(expect, actual)
    ieq(expect, actual)