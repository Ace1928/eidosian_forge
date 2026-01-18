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
@pytest.mark.parametrize('idx1', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
@pytest.mark.parametrize('idx2', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
def test_binops_index_pow(self, idx1, idx2):
    idx1 = idx1._rename('foo')
    idx2 = idx2._rename('bar')
    result = pow(idx1, idx2)
    expected = pow(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
    tm.assert_index_equal(result, expected, exact='equiv')