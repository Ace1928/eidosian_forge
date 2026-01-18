from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.dedup import duplicates, unique, conflicts, distinct, \
def test_conflicts_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    actual = conflicts(table, key='foo')
    ieq(expect, actual)