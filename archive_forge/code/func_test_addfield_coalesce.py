from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfield_coalesce():
    table = (('foo', 'bar', 'baz', 'quux'), ('M', 12, 23, 44), ('F', None, 23, 11), ('-', None, None, 42))
    result = addfield(table, 'spong', coalesce('bar', 'baz', 'quux'))
    expect = (('foo', 'bar', 'baz', 'quux', 'spong'), ('M', 12, 23, 44, 12), ('F', None, 23, 11, 23), ('-', None, None, 42, 42))
    ieq(expect, result)
    ieq(expect, result)
    result = addfield(table, 'spong', coalesce(1, 2, 3))
    expect = (('foo', 'bar', 'baz', 'quux', 'spong'), ('M', 12, 23, 44, 12), ('F', None, 23, 11, 23), ('-', None, None, 42, 42))
    ieq(expect, result)
    ieq(expect, result)