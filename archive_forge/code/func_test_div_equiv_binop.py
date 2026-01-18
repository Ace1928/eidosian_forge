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
def test_div_equiv_binop(self):
    first = Series([1, 0], name='first')
    second = Series([-0.01, -0.02], name='second')
    expected = Series([-0.01, -np.inf])
    result = second.div(first)
    tm.assert_series_equal(result, expected, check_names=False)
    result = second / first
    tm.assert_series_equal(result, expected)