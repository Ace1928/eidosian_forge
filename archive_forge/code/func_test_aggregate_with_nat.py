from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('func, fill_value', [('min', np.nan), ('max', np.nan), ('sum', 0), ('prod', 1), ('count', 0)])
def test_aggregate_with_nat(func, fill_value):
    n = 20
    data = np.random.default_rng(2).standard_normal((n, 4)).astype('int64')
    normal_df = DataFrame(data, columns=['A', 'B', 'C', 'D'])
    normal_df['key'] = [1, 2, np.nan, 4, 5] * 4
    dt_df = DataFrame(data, columns=['A', 'B', 'C', 'D'])
    dt_df['key'] = Index([datetime(2013, 1, 1), datetime(2013, 1, 2), pd.NaT, datetime(2013, 1, 4), datetime(2013, 1, 5)] * 4, dtype='M8[ns]')
    normal_grouped = normal_df.groupby('key')
    dt_grouped = dt_df.groupby(Grouper(key='key', freq='D'))
    normal_result = getattr(normal_grouped, func)()
    dt_result = getattr(dt_grouped, func)()
    pad = DataFrame([[fill_value] * 4], index=[3], columns=['A', 'B', 'C', 'D'])
    expected = pd.concat([normal_result, pad])
    expected = expected.sort_index()
    dti = date_range(start='2013-01-01', freq='D', periods=5, name='key', unit=dt_df['key']._values.unit)
    expected.index = dti._with_freq(None)
    tm.assert_frame_equal(expected, dt_result)
    assert dt_result.index.name == 'key'