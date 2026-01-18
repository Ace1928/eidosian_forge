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
def test_divmod_scalar(self, numeric_idx):
    idx = numeric_idx
    result = divmod(idx, 2)
    with np.errstate(all='ignore'):
        div, mod = divmod(idx.values, 2)
    expected = (Index(div), Index(mod))
    for r, e in zip(result, expected):
        tm.assert_index_equal(r, e)