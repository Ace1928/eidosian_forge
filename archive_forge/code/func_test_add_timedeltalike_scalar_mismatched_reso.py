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
@pytest.mark.parametrize('scalar', [timedelta(hours=2), pd.Timedelta(hours=2), np.timedelta64(2, 'h'), np.timedelta64(2 * 3600 * 1000, 'ms'), pd.offsets.Minute(120), pd.offsets.Hour(2)])
def test_add_timedeltalike_scalar_mismatched_reso(self, dta_dti, scalar):
    dta, dti = dta_dti
    td = pd.Timedelta(scalar)
    exp_unit = tm.get_finest_unit(dta.unit, td.unit)
    expected = (dti + td)._data.as_unit(exp_unit)
    result = dta + scalar
    tm.assert_extension_array_equal(result, expected)
    result = scalar + dta
    tm.assert_extension_array_equal(result, expected)
    expected = (dti - td)._data.as_unit(exp_unit)
    result = dta - scalar
    tm.assert_extension_array_equal(result, expected)