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
def test_getitem_ix_boolean_duplicates_multiple(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=['foo', 'foo', 'bar', 'baz', 'bar'])
    result = df.loc[['bar']]
    exp = df.iloc[[2, 4]]
    tm.assert_frame_equal(result, exp)
    result = df.loc[df[1] > 0]
    exp = df[df[1] > 0]
    tm.assert_frame_equal(result, exp)
    result = df.loc[df[0] > 0]
    exp = df[df[0] > 0]
    tm.assert_frame_equal(result, exp)