from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.dedup import duplicates, unique, conflicts, distinct, \
def test_duplicates_wholerow():
    table = (('foo', 'bar', 'baz'), ('A', 1, 2), ('B', '2', '3.4'), ('B', '2', '3.4'), ('D', 4, 12.3))
    result = duplicates(table)
    expectation = (('foo', 'bar', 'baz'), ('B', '2', '3.4'), ('B', '2', '3.4'))
    ieq(expectation, result)