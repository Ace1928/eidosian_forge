import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
def test_group_var_large_inputs(self):
    prng = np.random.default_rng(2)
    out = np.array([[np.nan]], dtype=self.dtype)
    counts = np.array([0], dtype='int64')
    values = (prng.random(10 ** 6) + 10 ** 12).astype(self.dtype)
    values.shape = (10 ** 6, 1)
    labels = np.zeros(10 ** 6, dtype='intp')
    self.algo(out, counts, values, labels)
    assert counts[0] == 10 ** 6
    tm.assert_almost_equal(out[0, 0], 1.0 / 12, rtol=0.0005)