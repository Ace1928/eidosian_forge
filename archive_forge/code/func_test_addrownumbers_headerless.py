from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addrownumbers_headerless():
    """Adds a column row if there is none."""
    table = ()
    expect = (('id',),)
    actual = addrownumbers(table, field='id')
    ieq(expect, actual)
    ieq(expect, actual)