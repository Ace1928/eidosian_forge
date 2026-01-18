from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_loc_nat(self):
    index = DatetimeIndex(['1/3/2000', 'NaT'])
    assert index.get_loc(pd.NaT) == 1
    assert index.get_loc(None) == 1
    assert index.get_loc(np.nan) == 1
    assert index.get_loc(pd.NA) == 1
    assert index.get_loc(np.datetime64('NaT')) == 1
    with pytest.raises(KeyError, match='NaT'):
        index.get_loc(np.timedelta64('NaT'))