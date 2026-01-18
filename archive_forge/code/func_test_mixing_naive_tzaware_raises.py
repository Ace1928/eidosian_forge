import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize('meth', [DatetimeArray._from_sequence, pd.to_datetime, pd.DatetimeIndex])
def test_mixing_naive_tzaware_raises(self, meth):
    arr = np.array([pd.Timestamp('2000'), pd.Timestamp('2000', tz='CET')])
    msg = 'Cannot mix tz-aware with tz-naive values|Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True'
    for obj in [arr, arr[::-1]]:
        with pytest.raises(ValueError, match=msg):
            meth(obj)