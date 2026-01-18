import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_apply_empty_series(engine_and_raw):
    engine, raw = engine_and_raw
    ser = Series([], dtype=np.float64)
    tm.assert_series_equal(ser, ser.expanding().apply(lambda x: x.mean(), raw=raw, engine=engine))