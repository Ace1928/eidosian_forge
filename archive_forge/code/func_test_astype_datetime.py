import numpy as np
import pytest
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]'])
def test_astype_datetime(dtype):
    arr = period_array(['2000', '2001', None], freq='D')
    if dtype == 'timedelta64[ns]':
        with pytest.raises(TypeError, match=dtype[:-4]):
            arr.astype(dtype)
    else:
        result = arr.astype(dtype)
        expected = pd.DatetimeIndex(['2000', '2001', pd.NaT], dtype=dtype)._data
        tm.assert_datetime_array_equal(result, expected)