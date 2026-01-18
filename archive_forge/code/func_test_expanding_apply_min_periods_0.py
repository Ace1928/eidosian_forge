import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_apply_min_periods_0(engine_and_raw):
    engine, raw = engine_and_raw
    s = Series([None, None, None])
    result = s.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
    expected = Series([1.0, 2.0, 3.0])
    tm.assert_series_equal(result, expected)