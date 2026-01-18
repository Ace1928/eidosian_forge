from datetime import timedelta
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isna_for_inf(self):
    s = Series(['a', np.inf, np.nan, pd.NA, 1.0])
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.use_inf_as_na', True):
            r = s.isna()
            dr = s.dropna()
    e = Series([False, True, True, True, False])
    de = Series(['a', 1.0], index=[0, 4])
    tm.assert_series_equal(r, e)
    tm.assert_series_equal(dr, de)