import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('expected_data, expected_index, axis', [[[np.nan, np.nan], range(2), 1], [[], [], 0]])
def test_datelike_numeric_only(self, expected_data, expected_index, axis):
    df = DataFrame({'a': pd.to_datetime(['2010', '2011']), 'b': [0, 5], 'c': pd.to_datetime(['2011', '2012'])})
    result = df[['a', 'c']].quantile(0.5, axis=axis, numeric_only=True)
    expected = Series(expected_data, name=0.5, index=Index(expected_index), dtype=np.float64)
    tm.assert_series_equal(result, expected)