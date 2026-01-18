import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concatlike_common_period_diff_freq_to_object(self):
    pi1 = pd.PeriodIndex(['2011-01', '2011-02'], freq='M')
    pi2 = pd.PeriodIndex(['2012-01-01', '2012-02-01'], freq='D')
    exp = Index([pd.Period('2011-01', freq='M'), pd.Period('2011-02', freq='M'), pd.Period('2012-01-01', freq='D'), pd.Period('2012-02-01', freq='D')], dtype=object)
    res = pi1.append(pi2)
    tm.assert_index_equal(res, exp)
    ps1 = Series(pi1)
    ps2 = Series(pi2)
    res = ps1._append(ps2)
    tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))
    res = pd.concat([ps1, ps2])
    tm.assert_series_equal(res, Series(exp, index=[0, 1, 0, 1]))