import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('arg,expected_bins', [[timedelta_range('1day', periods=3), TimedeltaIndex(['1 days', '2 days', '3 days'])], [date_range('20180101', periods=3), DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'])]])
def test_date_like_qcut_bins(arg, expected_bins):
    ser = Series(arg)
    result, result_bins = qcut(ser, 2, retbins=True)
    tm.assert_index_equal(result_bins, expected_bins)