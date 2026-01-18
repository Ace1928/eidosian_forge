from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfields():
    table = (('foo', 'bar'), ('M', 12), ('F', 34), ('-', 56))
    result = addfields(table, [('baz', 42), ('qux', lambda row: '%s,%s' % (row.foo, row.bar)), ('fiz', lambda rec: rec['bar'] * 2, 0)])
    expectation = (('fiz', 'foo', 'bar', 'baz', 'qux'), (24, 'M', 12, 42, 'M,12'), (68, 'F', 34, 42, 'F,34'), (112, '-', 56, 42, '-,56'))
    ieq(expectation, result)
    ieq(expectation, result)