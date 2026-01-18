from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_create_index_existing_name(self, simple_index):
    expected = simple_index
    if not isinstance(expected, MultiIndex):
        expected.name = 'foo'
        result = Index(expected)
        tm.assert_index_equal(result, expected)
        result = Index(expected, name='bar')
        expected.name = 'bar'
        tm.assert_index_equal(result, expected)
    else:
        expected.names = ['foo', 'bar']
        result = Index(expected)
        tm.assert_index_equal(result, Index(Index([('foo', 'one'), ('foo', 'two'), ('bar', 'one'), ('baz', 'two'), ('qux', 'one'), ('qux', 'two')], dtype='object'), names=['foo', 'bar']))
        result = Index(expected, names=['A', 'B'])
        tm.assert_index_equal(result, Index(Index([('foo', 'one'), ('foo', 'two'), ('bar', 'one'), ('baz', 'two'), ('qux', 'one'), ('qux', 'two')], dtype='object'), names=['A', 'B']))