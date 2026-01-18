import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_read_only(self):
    arr = np.array([1, 2, 3])
    arr.setflags(write=False)
    df = DataFrame([1, 2, 3])
    result = df.isin(arr)
    expected = DataFrame([True, True, True])
    tm.assert_frame_equal(result, expected)