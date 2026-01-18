import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_explode_scalars_can_ignore_index():
    s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    result = s.explode(ignore_index=True)
    expected = pd.Series([1, 2, 3])
    tm.assert_series_equal(result, expected)