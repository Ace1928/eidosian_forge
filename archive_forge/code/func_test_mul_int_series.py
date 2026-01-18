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
def test_mul_int_series(self, numeric_idx):
    idx = numeric_idx
    didx = idx * idx
    arr_dtype = 'uint64' if idx.dtype == np.uint64 else 'int64'
    result = idx * Series(np.arange(5, dtype=arr_dtype))
    tm.assert_series_equal(result, Series(didx))