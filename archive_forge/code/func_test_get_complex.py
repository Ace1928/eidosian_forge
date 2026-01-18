from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('idx, exp', [[2, [3, 3, np.nan, 'b']], [-1, [3, 3, np.nan, np.nan]]])
def test_get_complex(idx, exp):
    ser = Series([(1, 2, 3), [1, 2, 3], {1, 2, 3}, {1: 'a', 2: 'b', 3: 'c'}])
    result = ser.str.get(idx)
    expected = Series(exp)
    tm.assert_series_equal(result, expected)