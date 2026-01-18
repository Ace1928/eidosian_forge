import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ndarray_compat(self):
    tsdf = DataFrame(np.random.default_rng(2).standard_normal((1000, 3)), columns=['A', 'B', 'C'], index=date_range('1/1/2000', periods=1000))

    def f(x):
        return x[x.idxmax()]
    result = tsdf.apply(f)
    expected = tsdf.max()
    tm.assert_series_equal(result, expected)