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
@pytest.mark.parametrize('val', [None, [None], pd.NA, [pd.NA]])
def test_iloc_setitem_string_list_na(self, val):
    df = DataFrame({'a': ['a', 'b', 'c']}, dtype='string')
    df.iloc[[0], :] = val
    expected = DataFrame({'a': [pd.NA, 'b', 'c']}, dtype='string')
    tm.assert_frame_equal(df, expected)