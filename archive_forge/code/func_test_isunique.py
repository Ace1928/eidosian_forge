from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.dedup import duplicates, unique, conflicts, distinct, \
def test_isunique():
    table = (('foo', 'bar'), ('a', 1), ('b',), ('b', 2), ('c', 3, True))
    assert not isunique(table, 'foo')
    assert isunique(table, 'bar')