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
def test_frame_operators_none_to_nan(self):
    df = pd.DataFrame({'a': ['a', None, 'b']})
    tm.assert_frame_equal(df + df, pd.DataFrame({'a': ['aa', np.nan, 'bb']}))