import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('op, args, targop', [('cumprod', (), lambda x: x.cumprod()), ('cumsum', (), lambda x: x.cumsum()), ('shift', (-1,), lambda x: x.shift(-1)), ('shift', (1,), lambda x: x.shift())])
def test_cython_transform_series(op, args, targop):
    s = Series(np.random.default_rng(2).standard_normal(1000))
    s_missing = s.copy()
    s_missing.iloc[2:10] = np.nan
    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    for data in [s, s_missing]:
        expected = data.groupby(labels).transform(targop)
        tm.assert_series_equal(expected, data.groupby(labels).transform(op, *args))
        tm.assert_series_equal(expected, getattr(data.groupby(labels), op)(*args))