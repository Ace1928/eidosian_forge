from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_setitem_pure_position_based(self):
    ser1 = Series([1, 2, 3])
    ser2 = Series([4, 5, 6], index=[1, 0, 2])
    ser1.iloc[1:3] = ser2.iloc[1:3]
    expected = Series([1, 5, 6])
    tm.assert_series_equal(ser1, expected)