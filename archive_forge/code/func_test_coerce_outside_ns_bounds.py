from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('invalid_date', [date(1000, 1, 1), datetime(1000, 1, 1), '1000-01-01', 'Jan 1, 1000', np.datetime64('1000-01-01')])
@pytest.mark.parametrize('errors', ['coerce', 'raise'])
def test_coerce_outside_ns_bounds(invalid_date, errors):
    arr = np.array([invalid_date], dtype='object')
    kwargs = {'values': arr, 'errors': errors}
    if errors == 'raise':
        msg = '^Out of bounds nanosecond timestamp: .*, at position 0$'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            tslib.array_to_datetime(**kwargs)
    else:
        result, _ = tslib.array_to_datetime(**kwargs)
        expected = np.array([iNaT], dtype='M8[ns]')
        tm.assert_numpy_array_equal(result, expected)