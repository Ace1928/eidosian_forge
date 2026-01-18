from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfieldusingcontext():
    table1 = (('foo', 'bar'), ('A', 1), ('B', 4), ('C', 5), ('D', 9))
    expect = (('foo', 'bar', 'baz', 'quux'), ('A', 1, None, 3), ('B', 4, 3, 1), ('C', 5, 1, 4), ('D', 9, 4, None))

    def upstream(prv, cur, nxt):
        if prv is None:
            return None
        else:
            return cur.bar - prv.bar

    def downstream(prv, cur, nxt):
        if nxt is None:
            return None
        else:
            return nxt.bar - cur.bar
    table2 = addfieldusingcontext(table1, 'baz', upstream)
    table3 = addfieldusingcontext(table2, 'quux', downstream)
    ieq(expect, table3)
    ieq(expect, table3)