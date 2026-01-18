import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
def test_group_var_generic_2d_all_finite(self):
    prng = np.random.default_rng(2)
    out = (np.nan * np.ones((5, 2))).astype(self.dtype)
    counts = np.zeros(5, dtype='int64')
    values = 10 * prng.random((10, 2)).astype(self.dtype)
    labels = np.tile(np.arange(5), (2,)).astype('intp')
    expected_out = np.std(values.reshape(2, 5, 2), ddof=1, axis=0) ** 2
    expected_counts = counts + 2
    self.algo(out, counts, values, labels)
    assert np.allclose(out, expected_out, self.rtol)
    tm.assert_numpy_array_equal(counts, expected_counts)