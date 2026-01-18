import numpy as np
import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
@pytest.mark.xfail(reason='isna is not defined for MultiIndex')
def test_hasnans_isnans(idx):
    index = idx.copy()
    expected = np.array([False] * len(index), dtype=bool)
    tm.assert_numpy_array_equal(index._isnan, expected)
    assert index.hasnans is False
    index = idx.copy()
    values = index.values
    values[1] = np.nan
    index = type(idx)(values)
    expected = np.array([False] * len(index), dtype=bool)
    expected[1] = True
    tm.assert_numpy_array_equal(index._isnan, expected)
    assert index.hasnans is True