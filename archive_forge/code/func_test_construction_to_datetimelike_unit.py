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
@pytest.mark.parametrize('arr_dtype', [np.int64, np.float64])
@pytest.mark.parametrize('kind', ['M', 'm'])
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's', 'h', 'm', 'D'])
def test_construction_to_datetimelike_unit(self, arr_dtype, kind, unit):
    dtype = f'{kind}8[{unit}]'
    arr = np.array([1, 2, 3], dtype=arr_dtype)
    ser = Series(arr)
    result = ser.astype(dtype)
    expected = Series(arr.astype(dtype))
    if unit in ['ns', 'us', 'ms', 's']:
        assert result.dtype == dtype
        assert expected.dtype == dtype
    else:
        assert result.dtype == f'{kind}8[s]'
        assert expected.dtype == f'{kind}8[s]'
    tm.assert_series_equal(result, expected)