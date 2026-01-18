import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_calc_corr_small_numbers(self):
    df = DataFrame({'A': [1e-20, 2e-20, 3e-20], 'B': [1e-20, 2e-20, 3e-20]})
    result = df.corr()
    expected = DataFrame({'A': [1.0, 1.0], 'B': [1.0, 1.0]}, index=['A', 'B'])
    tm.assert_frame_equal(result, expected)