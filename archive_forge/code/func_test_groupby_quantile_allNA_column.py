import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Float64', 'Float32'])
def test_groupby_quantile_allNA_column(dtype):
    df = DataFrame({'x': [1, 1], 'y': [pd.NA] * 2}, dtype=dtype)
    result = df.groupby('x')['y'].quantile(0.5)
    expected = pd.Series([np.nan], dtype=dtype, index=Index([1.0], dtype=dtype), name='y')
    expected.index.name = 'x'
    tm.assert_series_equal(expected, result)