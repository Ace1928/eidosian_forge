import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('tz', [None, 'UTC', 'US/Pacific'])
def test_datetime_cut_roundtrip(tz, unit):
    ser = Series(date_range('20180101', periods=3, tz=tz, unit=unit))
    result, result_bins = cut(ser, 2, retbins=True)
    expected = cut(ser, result_bins)
    tm.assert_series_equal(result, expected)
    if unit == 's':
        expected_bins = DatetimeIndex(['2017-12-31 23:57:08', '2018-01-02 00:00:00', '2018-01-03 00:00:00'], dtype=f'M8[{unit}]')
    else:
        expected_bins = DatetimeIndex(['2017-12-31 23:57:07.200000', '2018-01-02 00:00:00', '2018-01-03 00:00:00'], dtype=f'M8[{unit}]')
    expected_bins = expected_bins.tz_localize(tz)
    tm.assert_index_equal(result_bins, expected_bins)