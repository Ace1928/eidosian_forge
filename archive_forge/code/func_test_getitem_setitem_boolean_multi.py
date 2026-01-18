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
def test_getitem_setitem_boolean_multi(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
    k1 = np.array([True, False, True])
    k2 = np.array([False, True])
    result = df.loc[k1, k2]
    expected = df.loc[[0, 2], [1]]
    tm.assert_frame_equal(result, expected)
    expected = df.copy()
    df.loc[np.array([True, False, True]), np.array([False, True])] = 5
    expected.loc[[0, 2], [1]] = 5
    tm.assert_frame_equal(df, expected)