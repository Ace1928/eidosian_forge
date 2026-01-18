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
def test_mul_float_series(self, numeric_idx):
    idx = numeric_idx
    rng5 = np.arange(5, dtype='float64')
    result = idx * Series(rng5 + 0.1)
    expected = Series(rng5 * (rng5 + 0.1))
    tm.assert_series_equal(result, expected)