import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_readonly(self):
    arr = np.random.default_rng(2).standard_normal((5, 2))
    arr.flags.writeable = False
    df = DataFrame(arr)
    result = df.diff()
    expected = DataFrame(np.array(df)).diff()
    tm.assert_frame_equal(result, expected)