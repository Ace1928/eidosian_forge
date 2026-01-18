from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('errors', ['ignore', 'coerce'])
def test_coerce_of_invalid_datetimes(errors):
    arr = np.array(['01-01-2013', 'not_a_date', '1'], dtype=object)
    kwargs = {'values': arr, 'errors': errors}
    if errors == 'ignore':
        result, _ = tslib.array_to_datetime(**kwargs)
        tm.assert_numpy_array_equal(result, arr)
    else:
        result, _ = tslib.array_to_datetime(arr, errors='coerce')
        expected = ['2013-01-01T00:00:00.000000000', iNaT, iNaT]
        tm.assert_numpy_array_equal(result, np.array(expected, dtype='M8[ns]'))