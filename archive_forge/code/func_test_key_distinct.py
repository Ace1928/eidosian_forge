from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.dedup import duplicates, unique, conflicts, distinct, \
def test_key_distinct():
    table = (('foo', 'bar', 'baz'), (None, None, None), ('A', 1, 2), ('B', '2', '3.4'), ('B', '2', '5'), ('D', 4, 12.3))
    result = distinct(table, key='foo')
    expect = (('foo', 'bar', 'baz'), (None, None, None), ('A', 1, 2), ('B', '2', '3.4'), ('D', 4, 12.3))
    ieq(expect, result)