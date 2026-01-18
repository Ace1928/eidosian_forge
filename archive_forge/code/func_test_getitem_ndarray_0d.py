import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_getitem_ndarray_0d(self):
    key = np.array(0)
    df = DataFrame([[1, 2], [3, 4]])
    result = df[key]
    expected = Series([1, 3], name=0)
    tm.assert_series_equal(result, expected)
    ser = Series([1, 2])
    result = ser[key]
    assert result == 1