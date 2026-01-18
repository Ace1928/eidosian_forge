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
def test_astype_to_sparse_dt64(self):
    dti = pd.date_range('2016-01-01', periods=4)
    dta = dti._data
    result = dta.astype('Sparse[datetime64[ns]]')
    assert result.dtype == 'Sparse[datetime64[ns]]'
    assert (result == dta).all()