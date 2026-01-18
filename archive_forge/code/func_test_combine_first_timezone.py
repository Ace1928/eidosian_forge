from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_timezone(self, unit):
    data1 = pd.to_datetime('20100101 01:01').tz_localize('UTC').as_unit(unit)
    df1 = DataFrame(columns=['UTCdatetime', 'abc'], data=data1, index=pd.date_range('20140627', periods=1))
    data2 = pd.to_datetime('20121212 12:12').tz_localize('UTC').as_unit(unit)
    df2 = DataFrame(columns=['UTCdatetime', 'xyz'], data=data2, index=pd.date_range('20140628', periods=1))
    res = df2[['UTCdatetime']].combine_first(df1)
    exp = DataFrame({'UTCdatetime': [pd.Timestamp('2010-01-01 01:01', tz='UTC'), pd.Timestamp('2012-12-12 12:12', tz='UTC')], 'abc': [pd.Timestamp('2010-01-01 01:01:00', tz='UTC'), pd.NaT]}, columns=['UTCdatetime', 'abc'], index=pd.date_range('20140627', periods=2, freq='D'), dtype=f'datetime64[{unit}, UTC]')
    assert res['UTCdatetime'].dtype == f'datetime64[{unit}, UTC]'
    assert res['abc'].dtype == f'datetime64[{unit}, UTC]'
    tm.assert_frame_equal(res, exp)