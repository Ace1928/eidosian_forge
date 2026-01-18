import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dropna_tz_aware_datetime(self):
    df = DataFrame()
    dt1 = datetime.datetime(2015, 1, 1, tzinfo=dateutil.tz.tzutc())
    dt2 = datetime.datetime(2015, 2, 2, tzinfo=dateutil.tz.tzutc())
    df['Time'] = [dt1]
    result = df.dropna(axis=0)
    expected = DataFrame({'Time': [dt1]})
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'Time': [dt1, None, np.nan, dt2]})
    result = df.dropna(axis=0)
    expected = DataFrame([dt1, dt2], columns=['Time'], index=[0, 3])
    tm.assert_frame_equal(result, expected)