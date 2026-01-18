from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_boolean_object(self, string_series):
    ser = string_series
    mask = ser > ser.median()
    omask = mask.astype(object)
    result = ser[omask]
    expected = ser[mask]
    tm.assert_series_equal(result, expected)
    s2 = ser.copy()
    cop = ser.copy()
    cop[omask] = 5
    s2[mask] = 5
    tm.assert_series_equal(cop, s2)
    omask[5:10] = np.nan
    msg = 'Cannot mask with non-boolean array containing NA / NaN values'
    with pytest.raises(ValueError, match=msg):
        ser[omask]
    with pytest.raises(ValueError, match=msg):
        ser[omask] = 5