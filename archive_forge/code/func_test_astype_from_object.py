import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_from_object(self):
    index = Index([1.0, np.nan, 0.2], dtype='object')
    result = index.astype(float)
    expected = Index([1.0, np.nan, 0.2], dtype=np.float64)
    assert result.dtype == expected.dtype
    tm.assert_index_equal(result, expected)