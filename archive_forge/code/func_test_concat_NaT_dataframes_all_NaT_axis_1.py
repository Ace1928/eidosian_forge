import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz1', [None, 'UTC'])
@pytest.mark.parametrize('tz2', [None, 'UTC'])
def test_concat_NaT_dataframes_all_NaT_axis_1(self, tz1, tz2):
    first = DataFrame(Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1))
    second = DataFrame(Series([pd.NaT]).dt.tz_localize(tz2), columns=[1])
    expected = DataFrame({0: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1), 1: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz2)})
    result = concat([first, second], axis=1)
    tm.assert_frame_equal(result, expected)