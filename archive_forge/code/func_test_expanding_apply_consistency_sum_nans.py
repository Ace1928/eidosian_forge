import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('f', [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_expanding_apply_consistency_sum_nans(request, all_data, min_periods, f):
    if f is np.sum:
        if not no_nans(all_data) and (not (all_na(all_data) and (not all_data.empty) and (min_periods > 0))):
            request.applymarker(pytest.mark.xfail(reason='np.sum has different behavior with NaNs'))
    expanding_f_result = all_data.expanding(min_periods=min_periods).sum()
    expanding_apply_f_result = all_data.expanding(min_periods=min_periods).apply(func=f, raw=True)
    tm.assert_equal(expanding_f_result, expanding_apply_f_result)