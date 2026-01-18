from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('series, func, expected', chain(tm.get_cython_table_params(Series(dtype=np.float64), [('sum', 0), ('max', np.nan), ('min', np.nan), ('all', True), ('any', False), ('mean', np.nan), ('prod', 1), ('std', np.nan), ('var', np.nan), ('median', np.nan)]), tm.get_cython_table_params(Series([np.nan, 1, 2, 3]), [('sum', 6), ('max', 3), ('min', 1), ('all', True), ('any', True), ('mean', 2), ('prod', 6), ('std', 1), ('var', 1), ('median', 2)]), tm.get_cython_table_params(Series('a b c'.split()), [('sum', 'abc'), ('max', 'c'), ('min', 'a'), ('all', True), ('any', True)])))
def test_agg_cython_table_series(series, func, expected):
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match='is currently using Series.*'):
        result = series.agg(func)
    if is_number(expected):
        assert np.isclose(result, expected, equal_nan=True)
    else:
        assert result == expected