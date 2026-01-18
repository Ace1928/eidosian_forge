from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.filterwarnings('ignore:Downcasting object dtype arrays:FutureWarning')
@pytest.mark.parametrize('opname', ['any', 'all'])
def test_any_all_bool_frame(self, opname, bool_frame_with_na):
    frame = bool_frame_with_na.fillna(True)
    alternative = getattr(np, opname)
    f = getattr(frame, opname)

    def skipna_wrapper(x):
        nona = x.dropna().values
        return alternative(nona)

    def wrapper(x):
        return alternative(x.values)
    result0 = f(axis=0, skipna=False)
    result1 = f(axis=1, skipna=False)
    tm.assert_series_equal(result0, frame.apply(wrapper))
    tm.assert_series_equal(result1, frame.apply(wrapper, axis=1))
    result0 = f(axis=0)
    result1 = f(axis=1)
    tm.assert_series_equal(result0, frame.apply(skipna_wrapper))
    tm.assert_series_equal(result1, frame.apply(skipna_wrapper, axis=1), check_dtype=False)
    with pytest.raises(ValueError, match='No axis named 2'):
        f(axis=2)
    all_na = frame * np.nan
    r0 = getattr(all_na, opname)(axis=0)
    r1 = getattr(all_na, opname)(axis=1)
    if opname == 'any':
        assert not r0.any()
        assert not r1.any()
    else:
        assert r0.all()
        assert r1.all()