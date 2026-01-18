from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_nancorr_pearson(self):
    targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
    targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
    self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method='pearson')
    targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
    targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
    self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='pearson')