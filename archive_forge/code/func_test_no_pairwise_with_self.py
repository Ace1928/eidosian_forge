import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('f', [lambda x: x.expanding().cov(pairwise=False), lambda x: x.expanding().corr(pairwise=False), lambda x: x.rolling(window=3).cov(pairwise=False), lambda x: x.rolling(window=3).corr(pairwise=False), lambda x: x.ewm(com=3).cov(pairwise=False), lambda x: x.ewm(com=3).corr(pairwise=False)])
def test_no_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):
    result = f(pairwise_frames)
    tm.assert_index_equal(result.index, pairwise_frames.index)
    tm.assert_index_equal(result.columns, pairwise_frames.columns)
    expected = f(pairwise_target_frame)
    result = result.dropna().values
    expected = expected.dropna().values
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)