import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('method, expected', [['cumsum', pd.Series([0, 1, np.nan, 1], dtype=object)], ['cumprod', pd.Series([False, 0, np.nan, 0])], ['cummin', pd.Series([False, False, np.nan, False])], ['cummax', pd.Series([False, True, np.nan, True])]])
def test_cummethods_bool_in_object_dtype(self, method, expected):
    ser = pd.Series([False, True, np.nan, False])
    result = getattr(ser, method)()
    tm.assert_series_equal(result, expected)