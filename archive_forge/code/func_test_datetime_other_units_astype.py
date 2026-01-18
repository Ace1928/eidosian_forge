from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['datetime64[D]', 'datetime64[h]', 'datetime64[m]', 'datetime64[s]', 'datetime64[ms]', 'datetime64[us]', 'datetime64[ns]'])
def test_datetime_other_units_astype(self, dtype):
    idx = DatetimeIndex(['2011-01-01', 'NaT', '2011-01-02'])
    values = idx.values.astype(dtype)
    exp = np.array([False, True, False])
    tm.assert_numpy_array_equal(isna(values), exp)
    tm.assert_numpy_array_equal(notna(values), ~exp)
    exp = Series([False, True, False])
    s = Series(values)
    tm.assert_series_equal(isna(s), exp)
    tm.assert_series_equal(notna(s), ~exp)
    s = Series(values, dtype=object)
    tm.assert_series_equal(isna(s), exp)
    tm.assert_series_equal(notna(s), ~exp)