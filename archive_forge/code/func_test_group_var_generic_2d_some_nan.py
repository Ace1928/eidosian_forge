import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
def test_group_var_generic_2d_some_nan(self):
    prng = np.random.default_rng(2)
    out = (np.nan * np.ones((5, 2))).astype(self.dtype)
    counts = np.zeros(5, dtype='int64')
    values = 10 * prng.random((10, 2)).astype(self.dtype)
    values[:, 1] = np.nan
    labels = np.tile(np.arange(5), (2,)).astype('intp')
    expected_out = np.vstack([values[:, 0].reshape(5, 2, order='F').std(ddof=1, axis=1) ** 2, np.nan * np.ones(5)]).T.astype(self.dtype)
    expected_counts = counts + 2
    self.algo(out, counts, values, labels)
    tm.assert_almost_equal(out, expected_out, rtol=5e-07)
    tm.assert_numpy_array_equal(counts, expected_counts)