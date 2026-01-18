import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('func', [np.cumsum, np.cumprod])
def test_datetime_series(self, datetime_series, func):
    tm.assert_numpy_array_equal(func(datetime_series).values, func(np.array(datetime_series)), check_dtype=True)
    ts = datetime_series.copy()
    ts[::2] = np.nan
    result = func(ts)[1::2]
    expected = func(np.array(ts.dropna()))
    tm.assert_numpy_array_equal(result.values, expected, check_dtype=False)