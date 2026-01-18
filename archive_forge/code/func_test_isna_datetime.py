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
def test_isna_datetime(self):
    assert not isna(datetime.now())
    assert notna(datetime.now())
    idx = date_range('1/1/1990', periods=20)
    exp = np.ones(len(idx), dtype=bool)
    tm.assert_numpy_array_equal(notna(idx), exp)
    idx = np.asarray(idx)
    idx[0] = iNaT
    idx = DatetimeIndex(idx)
    mask = isna(idx)
    assert mask[0]
    exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
    tm.assert_numpy_array_equal(mask, exp)
    pidx = idx.to_period(freq='M')
    mask = isna(pidx)
    assert mask[0]
    exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
    tm.assert_numpy_array_equal(mask, exp)
    mask = isna(pidx[1:])
    exp = np.zeros(len(mask), dtype=bool)
    tm.assert_numpy_array_equal(mask, exp)