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
def test_mul_size_mismatch_raises(self, numeric_idx):
    idx = numeric_idx
    msg = 'operands could not be broadcast together'
    with pytest.raises(ValueError, match=msg):
        idx * idx[0:3]
    with pytest.raises(ValueError, match=msg):
        idx * np.array([1, 2])