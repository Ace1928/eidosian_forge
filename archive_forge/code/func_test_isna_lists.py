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
def test_isna_lists(self):
    result = isna([[False]])
    exp = np.array([[False]])
    tm.assert_numpy_array_equal(result, exp)
    result = isna([[1], [2]])
    exp = np.array([[False], [False]])
    tm.assert_numpy_array_equal(result, exp)
    result = isna(['foo', 'bar'])
    exp = np.array([False, False])
    tm.assert_numpy_array_equal(result, exp)
    result = isna(['foo', 'bar'])
    exp = np.array([False, False])
    tm.assert_numpy_array_equal(result, exp)
    result = isna([np.nan, 'world'])
    exp = np.array([True, False])
    tm.assert_numpy_array_equal(result, exp)