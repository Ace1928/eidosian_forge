import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_nested_lists():
    s = pd.Series([[[1, 2, 3]], [1, 2], 1])
    result = s.explode()
    expected = pd.Series([[1, 2, 3], 1, 2, 1], index=[0, 1, 1, 2])
    tm.assert_series_equal(result, expected)