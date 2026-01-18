import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('inplace', [True, False])
def test_clip_against_series(self, inplace):
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
    lb = Series(np.random.default_rng(2).standard_normal(1000))
    ub = lb + 1
    original = df.copy()
    clipped_df = df.clip(lb, ub, axis=0, inplace=inplace)
    if inplace:
        clipped_df = df
    for i in range(2):
        lb_mask = original.iloc[:, i] <= lb
        ub_mask = original.iloc[:, i] >= ub
        mask = ~lb_mask & ~ub_mask
        result = clipped_df.loc[lb_mask, i]
        tm.assert_series_equal(result, lb[lb_mask], check_names=False)
        assert result.name == i
        result = clipped_df.loc[ub_mask, i]
        tm.assert_series_equal(result, ub[ub_mask], check_names=False)
        assert result.name == i
        tm.assert_series_equal(clipped_df.loc[mask, i], df.loc[mask, i])