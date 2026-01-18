from datetime import datetime
import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_dt64(self, unit):
    s0 = to_datetime(Series(['2010', np.nan])).dt.as_unit(unit)
    s1 = to_datetime(Series([np.nan, '2011'])).dt.as_unit(unit)
    rs = s0.combine_first(s1)
    xp = to_datetime(Series(['2010', '2011'])).dt.as_unit(unit)
    tm.assert_series_equal(rs, xp)
    s0 = to_datetime(Series(['2010', np.nan])).dt.as_unit(unit)
    s1 = Series([np.nan, '2011'])
    rs = s0.combine_first(s1)
    xp = Series([datetime(2010, 1, 1), '2011'], dtype='datetime64[ns]')
    tm.assert_series_equal(rs, xp)