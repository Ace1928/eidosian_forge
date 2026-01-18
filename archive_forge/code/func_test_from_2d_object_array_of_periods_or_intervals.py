import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_from_2d_object_array_of_periods_or_intervals(self):
    pi = pd.period_range('2016-04-05', periods=3)
    data = pi._data.astype(object).reshape(1, -1)
    df = DataFrame(data)
    assert df.shape == (1, 3)
    assert (df.dtypes == pi.dtype).all()
    assert (df == pi).all().all()
    ii = pd.IntervalIndex.from_breaks([3, 4, 5, 6])
    data2 = ii._data.astype(object).reshape(1, -1)
    df2 = DataFrame(data2)
    assert df2.shape == (1, 3)
    assert (df2.dtypes == ii.dtype).all()
    assert (df2 == ii).all().all()
    data3 = np.r_[data, data2, data, data2].T
    df3 = DataFrame(data3)
    expected = DataFrame({0: pi, 1: ii, 2: pi, 3: ii})
    tm.assert_frame_equal(df3, expected)