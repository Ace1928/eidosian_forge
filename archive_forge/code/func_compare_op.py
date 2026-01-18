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
def compare_op(series, other, op):
    left = np.abs(series) if op in (ops.rpow, operator.pow) else series
    right = np.abs(other) if op in (ops.rpow, operator.pow) else other
    cython_or_numpy = op(left, right)
    python = left.combine(right, op)
    if isinstance(other, Series) and (not other.index.equals(series.index)):
        python.index = python.index._with_freq(None)
    tm.assert_series_equal(cython_or_numpy, python)