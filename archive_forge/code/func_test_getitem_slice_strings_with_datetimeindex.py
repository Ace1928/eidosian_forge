from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_slice_strings_with_datetimeindex(self):
    idx = DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/3/2000', '1/4/2000'])
    ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)
    result = ts['1/2/2000':]
    expected = ts[1:]
    tm.assert_series_equal(result, expected)
    result = ts['1/2/2000':'1/3/2000']
    expected = ts[1:4]
    tm.assert_series_equal(result, expected)