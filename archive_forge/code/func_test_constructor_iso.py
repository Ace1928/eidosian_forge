from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
def test_constructor_iso(self):
    expected = timedelta_range('1s', periods=9, freq='s')
    durations = [f'P0DT0H0M{i}S' for i in range(1, 10)]
    result = to_timedelta(durations)
    tm.assert_index_equal(result, expected)