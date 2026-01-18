from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_sorting(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    ser = df.iloc[:, 0].sort_values()
    tm.assert_series_equal(ser, df.iloc[:, 0].sort_values())
    tm.assert_series_equal(ser, df[0].sort_values())