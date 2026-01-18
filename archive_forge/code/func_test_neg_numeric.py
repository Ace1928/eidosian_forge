from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('df,expected', [(pd.DataFrame({'a': [-1, 1]}), pd.DataFrame({'a': [1, -1]})), (pd.DataFrame({'a': [False, True]}), pd.DataFrame({'a': [True, False]})), (pd.DataFrame({'a': pd.Series(pd.to_timedelta([-1, 1]))}), pd.DataFrame({'a': pd.Series(pd.to_timedelta([1, -1]))}))])
def test_neg_numeric(self, df, expected):
    tm.assert_frame_equal(-df, expected)
    tm.assert_series_equal(-df['a'], expected['a'])