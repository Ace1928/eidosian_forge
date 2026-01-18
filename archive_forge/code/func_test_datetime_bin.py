import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('conv', [lambda v: Timestamp(v), lambda v: to_datetime(v), lambda v: np.datetime64(v), lambda v: Timestamp(v).to_pydatetime()])
def test_datetime_bin(conv):
    data = [np.datetime64('2012-12-13'), np.datetime64('2012-12-15')]
    bin_data = ['2012-12-12', '2012-12-14', '2012-12-16']
    expected = Series(IntervalIndex([Interval(Timestamp(bin_data[0]), Timestamp(bin_data[1])), Interval(Timestamp(bin_data[1]), Timestamp(bin_data[2]))])).astype(CategoricalDtype(ordered=True))
    bins = [conv(v) for v in bin_data]
    result = Series(cut(data, bins=bins))
    tm.assert_series_equal(result, expected)