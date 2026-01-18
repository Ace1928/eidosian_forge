from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dtype', ['boolean', 'Int64', 'UInt64', 'Float64'])
@pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('data,expected_data', [([0, 0, 0], [[False, False], [False, False]]), ([1, 1, 1], [[True, True], [True, True]]), ([pd.NA, pd.NA, pd.NA], [[pd.NA, pd.NA], [False, True]]), ([0, pd.NA, 0], [[pd.NA, False], [False, False]]), ([1, pd.NA, 1], [[True, pd.NA], [True, True]]), ([1, pd.NA, 0], [[True, False], [True, False]])])
def test_any_all_nullable_kleene_logic(self, bool_agg_func, skipna, data, dtype, expected_data):
    ser = Series(data, dtype=dtype)
    expected = expected_data[skipna][bool_agg_func == 'all']
    result = getattr(ser, bool_agg_func)(skipna=skipna)
    assert result is pd.NA and expected is pd.NA or result == expected