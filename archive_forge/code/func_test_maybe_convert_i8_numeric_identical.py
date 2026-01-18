from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('make_key', [IntervalIndex.from_breaks, lambda breaks: Interval(breaks[0], breaks[1]), lambda breaks: breaks[0]], ids=['IntervalIndex', 'Interval', 'scalar'])
def test_maybe_convert_i8_numeric_identical(self, make_key, any_real_numpy_dtype):
    breaks = np.arange(5, dtype=any_real_numpy_dtype)
    index = IntervalIndex.from_breaks(breaks)
    key = make_key(breaks)
    result = index._maybe_convert_i8(key)
    assert result is key