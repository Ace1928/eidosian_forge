from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def check_nancorr_nancov_1d(self, checkfun, targ0, targ1, **kwargs):
    res00 = checkfun(self.arr_float_1d, self.arr_float1_1d, **kwargs)
    res01 = checkfun(self.arr_float_1d, self.arr_float1_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
    tm.assert_almost_equal(targ0, res00)
    tm.assert_almost_equal(targ0, res01)
    res10 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, **kwargs)
    res11 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
    tm.assert_almost_equal(targ1, res10)
    tm.assert_almost_equal(targ1, res11)
    targ2 = np.nan
    res20 = checkfun(self.arr_nan_1d, self.arr_float1_1d, **kwargs)
    res21 = checkfun(self.arr_float_1d, self.arr_nan_1d, **kwargs)
    res22 = checkfun(self.arr_nan_1d, self.arr_nan_1d, **kwargs)
    res23 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, **kwargs)
    res24 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
    res25 = checkfun(self.arr_float_1d, self.arr_float1_1d, min_periods=len(self.arr_float_1d) + 1, **kwargs)
    tm.assert_almost_equal(targ2, res20)
    tm.assert_almost_equal(targ2, res21)
    tm.assert_almost_equal(targ2, res22)
    tm.assert_almost_equal(targ2, res23)
    tm.assert_almost_equal(targ2, res24)
    tm.assert_almost_equal(targ2, res25)