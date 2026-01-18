from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('series, func, expected', chain(tm.get_cython_table_params(Series(dtype=np.float64), [('cumprod', Series([], dtype=np.float64)), ('cumsum', Series([], dtype=np.float64))]), tm.get_cython_table_params(Series([np.nan, 1, 2, 3]), [('cumprod', Series([np.nan, 1, 2, 6])), ('cumsum', Series([np.nan, 1, 3, 6]))]), tm.get_cython_table_params(Series('a b c'.split()), [('cumsum', Series(['a', 'ab', 'abc']))])))
def test_agg_cython_table_transform_series(series, func, expected):
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match='is currently using Series.*'):
        result = series.agg(func)
    tm.assert_series_equal(result, expected)