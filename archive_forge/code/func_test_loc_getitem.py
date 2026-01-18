from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_loc_getitem(self, string_series, datetime_series):
    inds = string_series.index[[3, 4, 7]]
    tm.assert_series_equal(string_series.loc[inds], string_series.reindex(inds))
    tm.assert_series_equal(string_series.iloc[5::2], string_series[5::2])
    d1, d2 = datetime_series.index[[5, 15]]
    result = datetime_series.loc[d1:d2]
    expected = datetime_series.truncate(d1, d2)
    tm.assert_series_equal(result, expected)
    mask = string_series > string_series.median()
    tm.assert_series_equal(string_series.loc[mask], string_series[mask])
    assert datetime_series.loc[d1] == datetime_series[d1]
    assert datetime_series.loc[d2] == datetime_series[d2]