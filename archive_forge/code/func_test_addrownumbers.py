from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addrownumbers():
    table1 = (('foo', 'bar'), ('A', 9), ('C', 2), ('F', 1))
    expect = (('row', 'foo', 'bar'), (1, 'A', 9), (2, 'C', 2), (3, 'F', 1))
    actual = addrownumbers(table1)
    ieq(expect, actual)
    ieq(expect, actual)