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
def test_modulo_zero_int(self):
    with np.errstate(all='ignore'):
        s = Series([0, 1])
        result = s % 0
        expected = Series([np.nan, np.nan])
        tm.assert_series_equal(result, expected)
        result = 0 % s
        expected = Series([np.nan, 0.0])
        tm.assert_series_equal(result, expected)