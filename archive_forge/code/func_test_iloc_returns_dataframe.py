import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_iloc_returns_dataframe(simple_multiindex_dataframe):
    df = simple_multiindex_dataframe
    result = df.iloc[[0, 1]]
    expected = df.xs(4, drop_level=False)
    tm.assert_frame_equal(result, expected)