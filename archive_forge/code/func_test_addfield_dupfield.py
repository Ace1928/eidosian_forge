from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfield_dupfield():
    table = (('foo', 'foo'), ('M', 12), ('F', 34), ('-', 56))
    result = addfield(table, 'bar', 42)
    expectation = (('foo', 'foo', 'bar'), ('M', 12, 42), ('F', 34, 42), ('-', 56, 42))
    ieq(expectation, result)
    ieq(expectation, result)