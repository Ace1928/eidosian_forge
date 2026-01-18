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
def test_loc_set_nan_in_categorical_series(self, any_numeric_ea_dtype):
    srs = Series([1, 2, 3], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
    srs.loc[3] = np.nan
    expected = Series([1, 2, 3, np.nan], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
    tm.assert_series_equal(srs, expected)
    srs.loc[1] = np.nan
    expected = Series([1, np.nan, 3, np.nan], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
    tm.assert_series_equal(srs, expected)