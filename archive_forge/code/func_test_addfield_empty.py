from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfield_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar', 'baz'),)
    actual = addfield(table, 'baz', 42)
    ieq(expect, actual)
    ieq(expect, actual)