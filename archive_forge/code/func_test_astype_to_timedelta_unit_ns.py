import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['ns'])
def test_astype_to_timedelta_unit_ns(self, unit):
    dtype = f'm8[{unit}]'
    arr = np.array([[1, 2, 3]], dtype=dtype)
    df = DataFrame(arr)
    result = df.astype(dtype)
    expected = DataFrame(arr.astype(dtype))
    tm.assert_frame_equal(result, expected)