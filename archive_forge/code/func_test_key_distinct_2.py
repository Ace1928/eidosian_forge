from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.dedup import duplicates, unique, conflicts, distinct, \
def test_key_distinct_2():
    tbl = (('a', 'b'), ('x', '1'), ('x', '3'), ('y', '1'), (None, None))
    result = distinct(tbl, key='b')
    expect = (('a', 'b'), (None, None), ('x', '1'), ('x', '3'))
    ieq(expect, result)