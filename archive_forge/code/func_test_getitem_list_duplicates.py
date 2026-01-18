import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_list_duplicates(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=list('AABC'))
    df.columns.name = 'foo'
    result = df[['B', 'C']]
    assert result.columns.name == 'foo'
    expected = df.iloc[:, 2:]
    tm.assert_frame_equal(result, expected)