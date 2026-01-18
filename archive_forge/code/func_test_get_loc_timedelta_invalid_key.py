from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('key', [pd.Timedelta(0), pd.Timedelta(1), timedelta(0)])
def test_get_loc_timedelta_invalid_key(self, key):
    dti = date_range('1970-01-01', periods=10)
    msg = 'Cannot index DatetimeIndex with [Tt]imedelta'
    with pytest.raises(TypeError, match=msg):
        dti.get_loc(key)