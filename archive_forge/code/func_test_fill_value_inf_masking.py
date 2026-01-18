from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
def test_fill_value_inf_masking():
    df = pd.DataFrame({'A': [0, 1, 2], 'B': [1.1, None, 1.1]})
    other = pd.DataFrame({'A': [1.1, 1.2, 1.3]}, index=[0, 2, 3])
    result = df.rfloordiv(other, fill_value=1)
    expected = pd.DataFrame({'A': [np.inf, 1.0, 0.0, 1.0], 'B': [0.0, np.nan, 0.0, np.nan]})
    tm.assert_frame_equal(result, expected)