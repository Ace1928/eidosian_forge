import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('axis_name, axis_number', [pytest.param('rows', 0, id='rows_0'), pytest.param('index', 0, id='index_0'), pytest.param('columns', 1, id='columns_1')])
def test_interp_axis_names(self, axis_name, axis_number):
    data = {0: [0, np.nan, 6], 1: [1, np.nan, 7], 2: [2, 5, 8]}
    df = DataFrame(data, dtype=np.float64)
    result = df.interpolate(axis=axis_name, method='linear')
    expected = df.interpolate(axis=axis_number, method='linear')
    tm.assert_frame_equal(result, expected)