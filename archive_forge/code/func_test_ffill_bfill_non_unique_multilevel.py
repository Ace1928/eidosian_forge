import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func, expected_status', [('ffill', ['shrt', 'shrt', 'lng', np.nan, 'shrt', 'ntrl', 'ntrl']), ('bfill', ['shrt', 'lng', 'lng', 'shrt', 'shrt', 'ntrl', np.nan])])
def test_ffill_bfill_non_unique_multilevel(func, expected_status):
    date = pd.to_datetime(['2018-01-01', '2018-01-01', '2018-01-01', '2018-01-01', '2018-01-02', '2018-01-01', '2018-01-02'])
    symbol = ['MSFT', 'MSFT', 'MSFT', 'AAPL', 'AAPL', 'TSLA', 'TSLA']
    status = ['shrt', np.nan, 'lng', np.nan, 'shrt', 'ntrl', np.nan]
    df = DataFrame({'date': date, 'symbol': symbol, 'status': status})
    df = df.set_index(['date', 'symbol'])
    result = getattr(df.groupby('symbol')['status'], func)()
    index = MultiIndex.from_tuples(tuples=list(zip(*[date, symbol])), names=['date', 'symbol'])
    expected = Series(expected_status, index=index, name='status')
    tm.assert_series_equal(result, expected)