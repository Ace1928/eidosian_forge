from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
@pytest.mark.parametrize('data', [[False, None], [None, False], [False, np.nan], [np.nan, False]])
def test_any_all_object_dtype_missing(self, data, bool_agg_func):
    ser = Series(data)
    result = getattr(ser, bool_agg_func)(skipna=False)
    expected = bool_agg_func == 'any' and None not in data
    assert result == expected