from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import eq_
from petl.util.materialise import columns, facetcolumns
def test_columns():
    table = [['foo', 'bar'], ['a', 1], ['b', 2], ['b', 3]]
    cols = columns(table)
    eq_(['a', 'b', 'b'], cols['foo'])
    eq_([1, 2, 3], cols['bar'])