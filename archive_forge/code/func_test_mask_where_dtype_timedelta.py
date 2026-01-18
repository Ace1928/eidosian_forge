import numpy as np
from pandas import (
import pandas._testing as tm
def test_mask_where_dtype_timedelta():
    df = DataFrame([Timedelta(i, unit='d') for i in range(5)])
    expected = DataFrame(np.full(5, np.nan, dtype='timedelta64[ns]'))
    tm.assert_frame_equal(df.mask(df.notna()), expected)
    expected = DataFrame([np.nan, np.nan, np.nan, Timedelta('3 day'), Timedelta('4 day')])
    tm.assert_frame_equal(df.where(df > Timedelta(2, unit='d')), expected)