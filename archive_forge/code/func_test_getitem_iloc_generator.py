import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_iloc_generator(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    indexer = (x for x in [1, 2])
    result = df.iloc[indexer]
    expected = DataFrame({'a': [2, 3], 'b': [5, 6]}, index=[1, 2])
    tm.assert_frame_equal(result, expected)