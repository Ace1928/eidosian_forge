import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize(('pa_unit', 'pd_unit', 'pa_tz', 'pd_tz', 'data'), [('s', 's', 'UTC', 'UTC', EXTREME_VALUES), ('ms', 'ms', 'UTC', 'Europe/Berlin', EXTREME_VALUES), ('us', 'us', 'US/Eastern', 'UTC', EXTREME_VALUES), ('ns', 'ns', 'US/Central', 'Asia/Kolkata', EXTREME_VALUES), ('ns', 's', 'UTC', 'UTC', FINE_TO_COARSE_SAFE), ('us', 'ms', 'UTC', 'Europe/Berlin', FINE_TO_COARSE_SAFE), ('ms', 'us', 'US/Eastern', 'UTC', COARSE_TO_FINE_SAFE), ('s', 'ns', 'US/Central', 'Asia/Kolkata', COARSE_TO_FINE_SAFE)])
def test_from_arrow_with_different_units_and_timezones_with(pa_unit, pd_unit, pa_tz, pd_tz, data):
    pa = pytest.importorskip('pyarrow')
    pa_type = pa.timestamp(pa_unit, tz=pa_tz)
    arr = pa.array(data, type=pa_type)
    dtype = DatetimeTZDtype(unit=pd_unit, tz=pd_tz)
    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(data, dtype=f'M8[{pa_unit}, UTC]').astype(dtype, copy=False)
    tm.assert_extension_array_equal(result, expected)
    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)