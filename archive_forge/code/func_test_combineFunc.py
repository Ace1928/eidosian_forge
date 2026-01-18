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
def test_combineFunc(self, float_frame, mixed_float_frame):
    result = float_frame * 2
    tm.assert_numpy_array_equal(result.values, float_frame.values * 2)
    result = mixed_float_frame * 2
    for c, s in result.items():
        tm.assert_numpy_array_equal(s.values, mixed_float_frame[c].values * 2)
    _check_mixed_float(result, dtype={'C': None})
    result = DataFrame() * 2
    assert result.index.equals(DataFrame().index)
    assert len(result.columns) == 0