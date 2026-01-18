import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,arr_kwargs', [([1, 3, 4, 5], {'dtype': np.int64}), ([1.0, 3.0, 4.0, 5.0], {}), ([True, False, True, True], {})])
def test_list_numeric(data, arr_kwargs):
    result = to_numeric(data)
    expected = np.array(data, **arr_kwargs)
    tm.assert_numpy_array_equal(result, expected)