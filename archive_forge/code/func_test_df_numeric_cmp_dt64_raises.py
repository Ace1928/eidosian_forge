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
def test_df_numeric_cmp_dt64_raises(self, box_with_array, fixed_now_ts):
    ts = fixed_now_ts
    obj = np.array(range(5))
    obj = tm.box_expected(obj, box_with_array)
    assert_invalid_comparison(obj, ts, box_with_array)