import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_nan_object_dtype_nonmonotonic_nonunique(self):
    idx = Index(['foo', np.nan, None, 'foo', 1.0, None], dtype=object)
    res = idx.get_loc(np.nan)
    assert res == 1
    res = idx.get_loc(None)
    expected = np.array([False, False, True, False, False, True])
    tm.assert_numpy_array_equal(res, expected)
    with pytest.raises(KeyError, match='NaT'):
        idx.get_loc(NaT)