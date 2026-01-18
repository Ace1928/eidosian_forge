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
def test_divmod_series(self, numeric_idx):
    idx = numeric_idx
    other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2
    result = divmod(idx, Series(other))
    with np.errstate(all='ignore'):
        div, mod = divmod(idx.values, other)
    expected = (Series(div), Series(mod))
    for r, e in zip(result, expected):
        tm.assert_series_equal(r, e)