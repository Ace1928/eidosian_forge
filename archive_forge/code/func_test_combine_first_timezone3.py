from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_timezone3(self, unit):
    dts1 = pd.DatetimeIndex(['2011-01-01', 'NaT', '2011-01-03', '2011-01-04'], tz='US/Eastern').as_unit(unit)
    df1 = DataFrame({'DATE': dts1}, index=[1, 3, 5, 7])
    dts2 = pd.DatetimeIndex(['2012-01-01', '2012-01-02', '2012-01-03'], tz='US/Eastern').as_unit(unit)
    df2 = DataFrame({'DATE': dts2}, index=[2, 4, 5])
    res = df1.combine_first(df2)
    exp_dts = pd.DatetimeIndex(['2011-01-01', '2012-01-01', 'NaT', '2012-01-02', '2011-01-03', '2011-01-04'], tz='US/Eastern').as_unit(unit)
    exp = DataFrame({'DATE': exp_dts}, index=[1, 2, 3, 4, 5, 7])
    tm.assert_frame_equal(res, exp)