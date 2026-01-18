import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_axis(self):
    df = DataFrame([[1.0, 2.0], [3.0, 4.0]])
    tm.assert_frame_equal(df.diff(axis=1), DataFrame([[np.nan, 1.0], [np.nan, 1.0]]))
    tm.assert_frame_equal(df.diff(axis=0), DataFrame([[np.nan, np.nan], [2.0, 2.0]]))