import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_min_date_with_nans():
    dates = pd.to_datetime(Series(['2019-05-09', '2019-05-09', '2019-05-09']), format='%Y-%m-%d').dt.date
    df = DataFrame({'a': [np.nan, '1', np.nan], 'b': [0, 1, 1], 'c': dates})
    result = df.groupby('b', as_index=False)['c'].min()['c']
    expected = pd.to_datetime(Series(['2019-05-09', '2019-05-09'], name='c'), format='%Y-%m-%d').dt.date
    tm.assert_series_equal(result, expected)
    result = df.groupby('b')['c'].min()
    expected.index.name = 'b'
    tm.assert_series_equal(result, expected)