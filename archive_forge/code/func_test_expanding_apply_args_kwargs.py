import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_apply_args_kwargs(engine_and_raw):

    def mean_w_arg(x, const):
        return np.mean(x) + const
    engine, raw = engine_and_raw
    df = DataFrame(np.random.default_rng(2).random((20, 3)))
    expected = df.expanding().apply(np.mean, engine=engine, raw=raw) + 20.0
    result = df.expanding().apply(mean_w_arg, engine=engine, raw=raw, args=(20,))
    tm.assert_frame_equal(result, expected)
    result = df.expanding().apply(mean_w_arg, raw=raw, kwargs={'const': 20})
    tm.assert_frame_equal(result, expected)