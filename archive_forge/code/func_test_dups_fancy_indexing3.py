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
def test_dups_fancy_indexing3(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((9, 2)), index=[1, 1, 1, 2, 2, 2, 3, 3, 3], columns=['a', 'b'])
    expected = df.iloc[0:6]
    result = df.loc[[1, 2]]
    tm.assert_frame_equal(result, expected)
    expected = df
    result = df.loc[:, ['a', 'b']]
    tm.assert_frame_equal(result, expected)
    expected = df.iloc[0:6, :]
    result = df.loc[[1, 2], ['a', 'b']]
    tm.assert_frame_equal(result, expected)