import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arr_dtype', [np.int64, np.float64])
@pytest.mark.parametrize('dtype', ['M8', 'm8'])
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's', 'h', 'm', 'D'])
def test_astype_to_datetimelike_unit(self, arr_dtype, dtype, unit):
    dtype = f'{dtype}[{unit}]'
    arr = np.array([[1, 2, 3]], dtype=arr_dtype)
    df = DataFrame(arr)
    result = df.astype(dtype)
    expected = DataFrame(arr.astype(dtype))
    tm.assert_frame_equal(result, expected)