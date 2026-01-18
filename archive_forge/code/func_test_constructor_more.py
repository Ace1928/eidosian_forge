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
def test_constructor_more(self, float_frame):
    arr = np.random.default_rng(2).standard_normal(10)
    dm = DataFrame(arr, columns=['A'], index=np.arange(10))
    assert dm.values.ndim == 2
    arr = np.random.default_rng(2).standard_normal(0)
    dm = DataFrame(arr)
    assert dm.values.ndim == 2
    assert dm.values.ndim == 2
    dm = DataFrame(columns=['A', 'B'], index=np.arange(10))
    assert dm.values.shape == (10, 2)
    dm = DataFrame(columns=['A', 'B'])
    assert dm.values.shape == (0, 2)
    dm = DataFrame(index=np.arange(10))
    assert dm.values.shape == (10, 0)
    mat = np.array(['foo', 'bar'], dtype=object).reshape(2, 1)
    msg = "could not convert string to float: 'foo'"
    with pytest.raises(ValueError, match=msg):
        DataFrame(mat, index=[0, 1], columns=[0], dtype=float)
    dm = DataFrame(DataFrame(float_frame._series))
    tm.assert_frame_equal(dm, float_frame)
    dm = DataFrame({'A': np.ones(10, dtype=int), 'B': np.ones(10, dtype=np.float64)}, index=np.arange(10))
    assert len(dm.columns) == 2
    assert dm.values.dtype == np.float64