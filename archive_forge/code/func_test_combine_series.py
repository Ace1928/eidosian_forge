from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_combine_series(self, float_frame, mixed_float_frame, mixed_int_frame):
    series = float_frame.xs(float_frame.index[0])
    added = float_frame + series
    for key, s in added.items():
        tm.assert_series_equal(s, float_frame[key] + series[key])
    larger_series = series.to_dict()
    larger_series['E'] = 1
    larger_series = Series(larger_series)
    larger_added = float_frame + larger_series
    for key, s in float_frame.items():
        tm.assert_series_equal(larger_added[key], s + series[key])
    assert 'E' in larger_added
    assert np.isnan(larger_added['E']).all()
    added = mixed_float_frame + series
    assert np.all(added.dtypes == series.dtype)
    added = mixed_float_frame + series.astype('float32')
    _check_mixed_float(added, dtype={'C': None})
    added = mixed_float_frame + series.astype('float16')
    _check_mixed_float(added, dtype={'C': None})
    added = mixed_int_frame + (100 * series).astype('int64')
    _check_mixed_int(added, dtype={'A': 'int64', 'B': 'float64', 'C': 'int64', 'D': 'int64'})
    added = mixed_int_frame + (100 * series).astype('int32')
    _check_mixed_int(added, dtype={'A': 'int32', 'B': 'float64', 'C': 'int32', 'D': 'int64'})