import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import NaT
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize('tz', [None, 'US/Central'])
@pytest.mark.parametrize('skipna', [True, False])
def test_min_max_empty(self, skipna, tz):
    dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype('M8[ns]')
    arr = DatetimeArray._from_sequence([], dtype=dtype)
    result = arr.min(skipna=skipna)
    assert result is NaT
    result = arr.max(skipna=skipna)
    assert result is NaT