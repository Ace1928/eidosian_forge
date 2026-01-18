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
def test_loc_duplicates(self):
    trange = date_range(start=Timestamp(year=2017, month=1, day=1), end=Timestamp(year=2017, month=1, day=5))
    trange = trange.insert(loc=5, item=Timestamp(year=2017, month=1, day=5))
    df = DataFrame(0, index=trange, columns=['A', 'B'])
    bool_idx = np.array([False, False, False, False, False, True])
    df.loc[trange[bool_idx], 'A'] = 6
    expected = DataFrame({'A': [0, 0, 0, 0, 6, 6], 'B': [0, 0, 0, 0, 0, 0]}, index=trange)
    tm.assert_frame_equal(df, expected)
    df = DataFrame(0, index=trange, columns=['A', 'B'])
    df.loc[trange[bool_idx], 'A'] += 6
    tm.assert_frame_equal(df, expected)