from datetime import datetime
import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_mixed_timezone(self):
    uniform_tz = Series({pd.Timestamp('2019-05-01', tz='UTC'): 1.0})
    multi_tz = Series({pd.Timestamp('2019-05-01 01:00:00+0100', tz='Europe/London'): 2.0, pd.Timestamp('2019-05-02', tz='UTC'): 3.0})
    result = uniform_tz.combine_first(multi_tz)
    expected = Series([1.0, 3.0], index=pd.Index([pd.Timestamp('2019-05-01 00:00:00+00:00', tz='UTC'), pd.Timestamp('2019-05-02 00:00:00+00:00', tz='UTC')], dtype='object'))
    tm.assert_series_equal(result, expected)