import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('obj,expected', [(np.timedelta64(14, 'D'), 14 * 24 * 3600 * 1000000000.0), (Timedelta(minutes=-7), -7 * 60 * 1000000000.0), (Timedelta(minutes=-7).to_pytimedelta(), -7 * 60 * 1000000000.0), (Timedelta(seconds=1.234e-06), 1234), (Timedelta(seconds=1e-09, milliseconds=1e-05, microseconds=0.1), 111), (Timedelta(days=1, seconds=1e-09, milliseconds=1e-05, microseconds=0.1), 24 * 3600000000000.0 + 111), (offsets.Nano(125), 125)])
def test_delta_to_nanoseconds(obj, expected):
    result = delta_to_nanoseconds(obj)
    assert result == expected