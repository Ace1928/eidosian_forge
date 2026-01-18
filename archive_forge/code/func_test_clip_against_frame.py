import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('axis', [0, 1, None])
def test_clip_against_frame(self, axis):
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
    lb = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
    ub = lb + 1
    clipped_df = df.clip(lb, ub, axis=axis)
    lb_mask = df <= lb
    ub_mask = df >= ub
    mask = ~lb_mask & ~ub_mask
    tm.assert_frame_equal(clipped_df[lb_mask], lb[lb_mask])
    tm.assert_frame_equal(clipped_df[ub_mask], ub[ub_mask])
    tm.assert_frame_equal(clipped_df[mask], df[mask])