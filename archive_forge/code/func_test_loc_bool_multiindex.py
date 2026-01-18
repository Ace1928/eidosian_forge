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
@pytest.mark.parametrize('indexer', [True, (True,)])
@pytest.mark.parametrize('dtype', [bool, 'boolean'])
def test_loc_bool_multiindex(self, dtype, indexer):
    midx = MultiIndex.from_arrays([Series([True, True, False, False], dtype=dtype), Series([True, False, True, False], dtype=dtype)], names=['a', 'b'])
    df = DataFrame({'c': [1, 2, 3, 4]}, index=midx)
    with tm.maybe_produces_warning(PerformanceWarning, isinstance(indexer, tuple)):
        result = df.loc[indexer]
    expected = DataFrame({'c': [1, 2]}, index=Index([True, False], name='b', dtype=dtype))
    tm.assert_frame_equal(result, expected)