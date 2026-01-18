import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_iloc_two_dimensional_generator(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    indexer = (x for x in [1, 2])
    result = df.iloc[indexer, 1]
    expected = Series([5, 6], name='b', index=[1, 2])
    tm.assert_series_equal(result, expected)