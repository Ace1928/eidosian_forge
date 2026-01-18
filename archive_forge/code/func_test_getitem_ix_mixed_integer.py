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
def test_getitem_ix_mixed_integer(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)), index=[1, 10, 'C', 'E'], columns=[1, 2, 3])
    result = df.iloc[:-1]
    expected = df.loc[df.index[:-1]]
    tm.assert_frame_equal(result, expected)
    result = df.loc[[1, 10]]
    expected = df.loc[Index([1, 10])]
    tm.assert_frame_equal(result, expected)