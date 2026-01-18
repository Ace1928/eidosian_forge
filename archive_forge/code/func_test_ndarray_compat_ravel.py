import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ndarray_compat_ravel(self):
    s = Series(np.random.default_rng(2).standard_normal(10))
    with tm.assert_produces_warning(FutureWarning, match='ravel is deprecated'):
        result = s.ravel(order='F')
    tm.assert_almost_equal(result, s.values.ravel(order='F'))