import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('axis_name, axis_number', [('index', 0), ('columns', 1)])
def test_interp_string_axis(self, axis_name, axis_number):
    x = np.linspace(0, 100, 1000)
    y = np.sin(x)
    df = DataFrame(data=np.tile(y, (10, 1)), index=np.arange(10), columns=x).reindex(columns=x * 1.005)
    result = df.interpolate(method='linear', axis=axis_name)
    expected = df.interpolate(method='linear', axis=axis_number)
    tm.assert_frame_equal(result, expected)