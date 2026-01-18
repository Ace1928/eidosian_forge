import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import NaT
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_median_axis(self, arr1d):
    arr = arr1d
    assert arr.median(axis=0) == arr.median()
    assert arr.median(axis=0, skipna=False) is NaT
    msg = 'abs\\(axis\\) must be less than ndim'
    with pytest.raises(ValueError, match=msg):
        arr.median(axis=1)