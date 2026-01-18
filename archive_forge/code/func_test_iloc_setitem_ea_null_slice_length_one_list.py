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
@pytest.mark.parametrize('func', [list, Series, np.array])
def test_iloc_setitem_ea_null_slice_length_one_list(self, func):
    df = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
    df.iloc[:, func([0])] = 5
    expected = DataFrame({'a': [5, 5, 5]}, dtype='Int64')
    tm.assert_frame_equal(df, expected)