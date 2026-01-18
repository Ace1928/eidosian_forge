from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_loc_key_unit_mismatch_not_castable(self):
    dta = date_range('2000-01-01', periods=3)._data.astype('M8[s]')
    dti = DatetimeIndex(dta)
    key = dta[0].as_unit('ns') + pd.Timedelta(1)
    with pytest.raises(KeyError, match="Timestamp\\('2000-01-01 00:00:00.000000001'\\)"):
        dti.get_loc(key)
    assert key not in dti