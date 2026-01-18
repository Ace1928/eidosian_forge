from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
@pytest.mark.parametrize('values', [[np.datetime64('2012-01-01'), np.datetime64('2013-01-01')], ['2012-01-01', '2013-01-01']])
def test_constructor_sparse_datetime64(self, values):
    dtype = pd.SparseDtype('datetime64[ns]')
    result = Series(values, dtype=dtype)
    arr = pd.arrays.SparseArray(values, dtype=dtype)
    expected = Series(arr)
    tm.assert_series_equal(result, expected)