import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('func, exp', [('cummin', pd.Period('2012-1-1', freq='D')), ('cummax', pd.Period('2012-1-2', freq='D'))])
def test_cummin_cummax_period(self, func, exp):
    ser = pd.Series([pd.Period('2012-1-1', freq='D'), pd.NaT, pd.Period('2012-1-2', freq='D')])
    result = getattr(ser, func)(skipna=False)
    expected = pd.Series([pd.Period('2012-1-1', freq='D'), pd.NaT, pd.NaT])
    tm.assert_series_equal(result, expected)
    result = getattr(ser, func)(skipna=True)
    expected = pd.Series([pd.Period('2012-1-1', freq='D'), pd.NaT, exp])
    tm.assert_series_equal(result, expected)