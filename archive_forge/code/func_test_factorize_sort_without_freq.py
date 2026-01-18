from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_factorize_sort_without_freq():
    dta = DatetimeArray._from_sequence([0, 2, 1], dtype='M8[ns]')
    msg = 'call pd.factorize\\(obj, sort=True\\) instead'
    with pytest.raises(NotImplementedError, match=msg):
        dta.factorize(sort=True)
    tda = dta - dta[0]
    with pytest.raises(NotImplementedError, match=msg):
        tda.factorize(sort=True)