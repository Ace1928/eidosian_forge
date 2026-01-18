import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [int, 'int16', 'int32', 'int64'])
@pytest.mark.parametrize('non_finite', [np.inf, np.nan])
def test_cannot_cast_inf_to_int(self, non_finite, dtype):
    idx = Index([1, 2, non_finite], dtype=np.float64)
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    with pytest.raises(ValueError, match=msg):
        idx.astype(dtype)