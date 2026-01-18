from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('col_name', ['t2', 't2x', 't2q', 'T_2M', 't2p', 't2m', 't2m1', 'T2M'])
def test_agg_with_datetime_index_list_agg_func(col_name):
    df = DataFrame(list(range(200)), index=date_range(start='2017-01-01', freq='15min', periods=200, tz='Europe/Berlin'), columns=[col_name])
    result = df.resample('1d').aggregate(['mean'])
    expected = DataFrame([47.5, 143.5, 195.5], index=date_range(start='2017-01-01', freq='D', periods=3, tz='Europe/Berlin'), columns=pd.MultiIndex(levels=[[col_name], ['mean']], codes=[[0], [0]]))
    tm.assert_frame_equal(result, expected)