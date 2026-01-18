from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import eq_
from petl.util.materialise import columns, facetcolumns
def test_columns_empty():
    table = [('foo', 'bar')]
    cols = columns(table)
    eq_([], cols['foo'])
    eq_([], cols['bar'])