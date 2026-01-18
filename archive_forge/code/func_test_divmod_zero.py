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
def test_divmod_zero(self, zero, numeric_idx):
    idx = numeric_idx
    exleft = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
    exright = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    exleft = adjust_negative_zero(zero, exleft)
    result = divmod(idx, zero)
    tm.assert_index_equal(result[0], exleft)
    tm.assert_index_equal(result[1], exright)