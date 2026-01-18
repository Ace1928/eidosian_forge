from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_sort_multiindex_series():
    index = MultiIndex(levels=[[1, 2], [1, 2]], codes=[[0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0]], names=['a', 'b'])
    mseries = Series([0, 1, 2, 3, 4, 5], index=index)
    index = MultiIndex(levels=[[1, 2], [1, 2]], codes=[[0, 0, 1], [1, 0, 0]], names=['a', 'b'])
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=['a', 'b'], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=['a', 'b'], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())