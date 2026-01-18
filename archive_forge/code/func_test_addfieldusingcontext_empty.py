from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfieldusingcontext_empty():
    table = empty()
    expect = (('foo',),)

    def query(prv, cur, nxt):
        return 0
    actual = addfieldusingcontext(table, 'foo', query)
    ieq(expect, actual)
    ieq(expect, actual)