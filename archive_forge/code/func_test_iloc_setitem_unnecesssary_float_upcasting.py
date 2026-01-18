import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_iloc_setitem_unnecesssary_float_upcasting():
    df = DataFrame({0: np.array([1, 3], dtype=np.float32), 1: np.array([2, 4], dtype=np.float32), 2: ['a', 'b']})
    orig = df.copy()
    values = df[0].values.reshape(2, 1)
    df.iloc[:, 0:1] = values
    tm.assert_frame_equal(df, orig)