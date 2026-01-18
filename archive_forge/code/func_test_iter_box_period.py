import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_iter_box_period(self):
    vals = [pd.Period('2011-01-01', freq='M'), pd.Period('2011-01-02', freq='M')]
    s = Series(vals)
    assert s.dtype == 'Period[M]'
    for res, exp in zip(s, vals):
        assert isinstance(res, pd.Period)
        assert res.freq == 'ME'
        assert res == exp