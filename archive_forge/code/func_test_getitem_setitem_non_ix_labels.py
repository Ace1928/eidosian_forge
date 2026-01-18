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
def test_getitem_setitem_non_ix_labels(self):
    df = DataFrame(range(20), index=date_range('2020-01-01', periods=20))
    start, end = df.index[[5, 10]]
    result = df.loc[start:end]
    result2 = df[start:end]
    expected = df[5:11]
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)
    result = df.copy()
    result.loc[start:end] = 0
    result2 = df.copy()
    result2[start:end] = 0
    expected = df.copy()
    expected[5:11] = 0
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)