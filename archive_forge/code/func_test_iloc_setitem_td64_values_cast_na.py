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
@pytest.mark.parametrize('value', [None, NaT, np.nan])
def test_iloc_setitem_td64_values_cast_na(self, value):
    series = Series([0, 1, 2], dtype='timedelta64[ns]')
    series.iloc[0] = value
    expected = Series([NaT, 1, 2], dtype='timedelta64[ns]')
    tm.assert_series_equal(series, expected)