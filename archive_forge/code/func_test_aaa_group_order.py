from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
def test_aaa_group_order():
    n = 20
    data = np.random.default_rng(2).standard_normal((n, 4))
    df = DataFrame(data, columns=['A', 'B', 'C', 'D'])
    df['key'] = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3), datetime(2013, 1, 4), datetime(2013, 1, 5)] * 4
    grouped = df.groupby(Grouper(key='key', freq='D'))
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 1)), df[::5])
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 2)), df[1::5])
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 3)), df[2::5])
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 4)), df[3::5])
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 5)), df[4::5])