import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_clip_int_data_with_float_bound(self):
    df = DataFrame({'a': [1, 2, 3]})
    result = df.clip(lower=1.5)
    expected = DataFrame({'a': [1.5, 2.0, 3.0]})
    tm.assert_frame_equal(result, expected)