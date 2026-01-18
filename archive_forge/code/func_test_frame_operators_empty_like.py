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
@pytest.mark.parametrize('dtype', ('float', 'int64'))
def test_frame_operators_empty_like(self, dtype):
    frames = [pd.DataFrame(dtype=dtype), pd.DataFrame(columns=['A'], dtype=dtype), pd.DataFrame(index=[0], dtype=dtype)]
    for df in frames:
        assert (df + df).equals(df)
        tm.assert_frame_equal(df + df, df)