from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_dtype_datetime64_10(self):
    pydates = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)]
    dates = [np.datetime64(x) for x in pydates]
    ser = Series(dates)
    assert ser.dtype == 'M8[ns]'
    ser.iloc[0] = np.nan
    assert ser.dtype == 'M8[ns]'
    expected = Series(pydates, dtype='datetime64[ms]')
    result = Series(Series(dates).astype(np.int64) / 1000000, dtype='M8[ms]')
    tm.assert_series_equal(result, expected)
    result = Series(dates, dtype='datetime64[ms]')
    tm.assert_series_equal(result, expected)
    expected = Series([NaT, datetime(2013, 1, 2), datetime(2013, 1, 3)], dtype='datetime64[ns]')
    result = Series([np.nan] + dates[1:], dtype='datetime64[ns]')
    tm.assert_series_equal(result, expected)