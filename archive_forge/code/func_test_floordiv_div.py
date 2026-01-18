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
def test_floordiv_div(self):
    ser = Series([-1, 0, 1], name='first')
    result = ser // 0
    expected = Series([-np.inf, np.nan, np.inf], name='first')
    tm.assert_series_equal(result, expected)