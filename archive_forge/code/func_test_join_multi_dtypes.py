from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('d1', [np.int64, np.int32, np.intc, np.int16, np.int8, np.uint8])
@pytest.mark.parametrize('d2', [np.int64, np.float64, np.float32, np.float16])
def test_join_multi_dtypes(self, d1, d2):
    dtype1 = np.dtype(d1)
    dtype2 = np.dtype(d2)
    left = DataFrame({'k1': np.array([0, 1, 2] * 8, dtype=dtype1), 'k2': ['foo', 'bar'] * 12, 'v': np.array(np.arange(24), dtype=np.int64)})
    index = MultiIndex.from_tuples([(2, 'bar'), (1, 'foo')])
    right = DataFrame({'v2': np.array([5, 7], dtype=dtype2)}, index=index)
    result = left.join(right, on=['k1', 'k2'])
    expected = left.copy()
    if dtype2.kind == 'i':
        dtype2 = np.dtype('float64')
    expected['v2'] = np.array(np.nan, dtype=dtype2)
    expected.loc[(expected.k1 == 2) & (expected.k2 == 'bar'), 'v2'] = 5
    expected.loc[(expected.k1 == 1) & (expected.k2 == 'foo'), 'v2'] = 7
    tm.assert_frame_equal(result, expected)
    result = left.join(right, on=['k1', 'k2'], sort=True)
    expected.sort_values(['k1', 'k2'], kind='mergesort', inplace=True)
    tm.assert_frame_equal(result, expected)