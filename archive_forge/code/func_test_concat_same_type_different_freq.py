from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_concat_same_type_different_freq(self, unit):
    a = pd.date_range('2000', periods=2, freq='D', tz='US/Central', unit=unit)._data
    b = pd.date_range('2000', periods=2, freq='h', tz='US/Central', unit=unit)._data
    result = DatetimeArray._concat_same_type([a, b])
    expected = pd.to_datetime(['2000-01-01 00:00:00', '2000-01-02 00:00:00', '2000-01-01 00:00:00', '2000-01-01 01:00:00']).tz_localize('US/Central').as_unit(unit)._data
    tm.assert_datetime_array_equal(result, expected)