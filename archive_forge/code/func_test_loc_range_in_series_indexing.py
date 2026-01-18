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
@pytest.mark.parametrize('size', [5, 999999, 1000000])
def test_loc_range_in_series_indexing(self, size):
    s = Series(index=range(size), dtype=np.float64)
    s.loc[range(1)] = 42
    tm.assert_series_equal(s.loc[range(1)], Series(42.0, index=[0]))
    s.loc[range(2)] = 43
    tm.assert_series_equal(s.loc[range(2)], Series(43.0, index=[0, 1]))