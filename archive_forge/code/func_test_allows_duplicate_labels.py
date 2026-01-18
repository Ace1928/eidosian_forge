import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_allows_duplicate_labels():
    left = DataFrame()
    right = DataFrame().set_flags(allows_duplicate_labels=False)
    tm.assert_frame_equal(left, left)
    tm.assert_frame_equal(right, right)
    tm.assert_frame_equal(left, right, check_flags=False)
    tm.assert_frame_equal(right, left, check_flags=False)
    with pytest.raises(AssertionError, match='<Flags'):
        tm.assert_frame_equal(left, right)
    with pytest.raises(AssertionError, match='<Flags'):
        tm.assert_frame_equal(left, right)