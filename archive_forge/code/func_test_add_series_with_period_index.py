from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_add_series_with_period_index(self):
    rng = pd.period_range('1/1/2000', '1/1/2010', freq='Y')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    result = ts + ts[::2]
    expected = ts + ts
    expected.iloc[1::2] = np.nan
    tm.assert_series_equal(result, expected)
    result = ts + _permute(ts[::2])
    tm.assert_series_equal(result, expected)
    msg = 'Input has different freq=D from Period\\(freq=Y-DEC\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        ts + ts.asfreq('D', how='end')