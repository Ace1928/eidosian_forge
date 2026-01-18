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
@pytest.mark.parametrize('col', [{}, {'name': 'a'}])
def test_loc_setitem_reordering_with_all_true_indexer(self, col):
    n = 17
    df = DataFrame({**col, 'x': range(n), 'y': range(n)})
    expected = df.copy()
    df.loc[n * [True], ['x', 'y']] = df[['x', 'y']]
    tm.assert_frame_equal(df, expected)