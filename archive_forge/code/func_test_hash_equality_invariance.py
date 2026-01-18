from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
@pytest.mark.skip_ubsan
@pytest.mark.xfail(reason='pd.Timedelta violates the Python hash invariant (GH#44504).')
@given(st.integers(min_value=(-sys.maxsize - 1) // 500, max_value=sys.maxsize // 500))
def test_hash_equality_invariance(self, half_microseconds: int) -> None:
    nanoseconds = half_microseconds * 500
    pandas_timedelta = Timedelta(nanoseconds)
    numpy_timedelta = np.timedelta64(nanoseconds)
    assert pandas_timedelta != numpy_timedelta or hash(pandas_timedelta) == hash(numpy_timedelta)