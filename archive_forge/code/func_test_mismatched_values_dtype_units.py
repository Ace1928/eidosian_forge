import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_mismatched_values_dtype_units(self):
    arr = np.array([1, 2, 3], dtype='M8[s]')
    dtype = np.dtype('M8[ns]')
    msg = 'Values resolution does not match dtype.'
    depr_msg = 'DatetimeArray.__init__ is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match=msg):
            DatetimeArray(arr, dtype=dtype)
    dtype2 = DatetimeTZDtype(tz='UTC', unit='ns')
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match=msg):
            DatetimeArray(arr, dtype=dtype2)