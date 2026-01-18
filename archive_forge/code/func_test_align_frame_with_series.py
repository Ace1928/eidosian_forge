from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_frame_with_series(self, float_frame):
    idx = float_frame.index
    s = Series(range(len(idx)), index=idx)
    left, right = float_frame.align(s, axis=0)
    tm.assert_index_equal(left.index, float_frame.index)
    tm.assert_index_equal(right.index, float_frame.index)
    assert isinstance(right, Series)
    msg = "The 'broadcast_axis' keyword in DataFrame.align is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        left, right = float_frame.align(s, broadcast_axis=1)
    tm.assert_index_equal(left.index, float_frame.index)
    expected = {c: s for c in float_frame.columns}
    expected = DataFrame(expected, index=float_frame.index, columns=float_frame.columns)
    tm.assert_frame_equal(right, expected)