import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func, values', [('idxmin', ['1/1/2011'] * 2 + ['1/3/2011'] * 7 + ['1/10/2011']), ('idxmax', ['1/2/2011'] * 2 + ['1/9/2011'] * 7 + ['1/10/2011'])])
def test_groupby_transform_with_datetimes(func, values):
    dates = date_range('1/1/2011', periods=10, freq='D')
    stocks = DataFrame({'price': np.arange(10.0)}, index=dates)
    stocks['week_id'] = dates.isocalendar().week
    result = stocks.groupby(stocks['week_id'])['price'].transform(func)
    expected = Series(data=pd.to_datetime(values).as_unit('ns'), index=dates, name='price')
    tm.assert_series_equal(result, expected)