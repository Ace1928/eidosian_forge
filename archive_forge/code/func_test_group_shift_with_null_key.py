import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_group_shift_with_null_key():
    n_rows = 1200
    df = DataFrame([(i % 12, i % 3 if i % 3 else np.nan, i) for i in range(n_rows)], dtype=float, columns=['A', 'B', 'Z'], index=None)
    g = df.groupby(['A', 'B'])
    expected = DataFrame([i + 12 if i % 3 and i < n_rows - 12 else np.nan for i in range(n_rows)], dtype=float, columns=['Z'], index=None)
    result = g.shift(-1)
    tm.assert_frame_equal(result, expected)