from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_iloc_col_slice_view(self, using_array_manager, using_copy_on_write, warn_copy_on_write):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2))
    original = df.copy()
    subset = df.iloc[:, slice(4, 8)]
    if not using_array_manager and (not using_copy_on_write):
        assert np.shares_memory(df[8]._values, subset[8]._values)
        with tm.assert_cow_warning(warn_copy_on_write):
            subset.loc[:, 8] = 0.0
        assert (df[8] == 0).all()
        assert np.shares_memory(df[8]._values, subset[8]._values)
    else:
        if using_copy_on_write:
            assert np.shares_memory(df[8]._values, subset[8]._values)
        subset[8] = 0.0
        assert (subset[8] == 0).all()
        tm.assert_frame_equal(df, original)