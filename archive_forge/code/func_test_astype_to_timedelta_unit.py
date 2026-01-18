import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['us', 'ms', 's', 'h', 'm', 'D'])
def test_astype_to_timedelta_unit(self, unit):
    dtype = f'm8[{unit}]'
    arr = np.array([[1, 2, 3]], dtype=dtype)
    df = DataFrame(arr)
    ser = df.iloc[:, 0]
    tdi = Index(ser)
    tda = tdi._values
    if unit in ['us', 'ms', 's']:
        assert (df.dtypes == dtype).all()
        result = df.astype(dtype)
    else:
        assert (df.dtypes == 'm8[s]').all()
        msg = f"Cannot convert from timedelta64\\[s\\] to timedelta64\\[{unit}\\]. Supported resolutions are 's', 'ms', 'us', 'ns'"
        with pytest.raises(ValueError, match=msg):
            df.astype(dtype)
        with pytest.raises(ValueError, match=msg):
            ser.astype(dtype)
        with pytest.raises(ValueError, match=msg):
            tdi.astype(dtype)
        with pytest.raises(ValueError, match=msg):
            tda.astype(dtype)
        return
    result = df.astype(dtype)
    expected = df
    tm.assert_frame_equal(result, expected)