from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_nanvar_ddof(self):
    n = 5
    samples = self.prng.uniform(size=(10000, n + 1))
    samples[:, -1] = np.nan
    variance_0 = nanops.nanvar(samples, axis=1, skipna=True, ddof=0).mean()
    variance_1 = nanops.nanvar(samples, axis=1, skipna=True, ddof=1).mean()
    variance_2 = nanops.nanvar(samples, axis=1, skipna=True, ddof=2).mean()
    var = 1.0 / 12
    tm.assert_almost_equal(variance_1, var, rtol=0.01)
    tm.assert_almost_equal(variance_0, (n - 1.0) / n * var, rtol=0.01)
    tm.assert_almost_equal(variance_2, (n - 1.0) / (n - 2.0) * var, rtol=0.01)