import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx,target,expected', [([np.nan, 'var1', np.nan], [np.nan], np.array([0, 2], dtype=np.intp)), ([np.nan, 'var1', np.nan], [np.nan, 'var1'], np.array([0, 2, 1], dtype=np.intp)), (np.array([np.nan, 'var1', np.nan], dtype=object), [np.nan], np.array([0, 2], dtype=np.intp)), (DatetimeIndex(['2020-08-05', NaT, NaT]), [NaT], np.array([1, 2], dtype=np.intp)), (['a', 'b', 'a', np.nan], [np.nan], np.array([3], dtype=np.intp)), (np.array(['b', np.nan, float('NaN'), 'b'], dtype=object), Index([np.nan], dtype=object), np.array([1, 2], dtype=np.intp))])
def test_get_indexer_non_unique_multiple_nans(idx, target, expected):
    axis = Index(idx)
    actual = axis.get_indexer_for(target)
    tm.assert_numpy_array_equal(actual, expected)