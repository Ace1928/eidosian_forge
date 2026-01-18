import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['timedelta64[ns]', 'datetime64[ns, UTC]', 'Period[D]'])
def test_assert_frame_equal_datetime_like_dtype_mismatch(dtype):
    df1 = DataFrame({'a': []}, dtype=dtype)
    df2 = DataFrame({'a': []})
    tm.assert_frame_equal(df1, df2, check_dtype=False)