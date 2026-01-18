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
@pytest.mark.parametrize('first, second, expected', [(pd.DataFrame({'x': [1, 2, 3]}, index=list('ABC')), pd.DataFrame({'x': [2, 2, 2]}, index=list('ABD')), pd.DataFrame({'x': [3.0, 4.0, np.nan, np.nan]}, index=list('ABCD'))), (pd.DataFrame({'x': [1, 2, 3]}, index=list('ABC')), pd.DataFrame({'x': [2, 2, 2, 2]}, index=list('ABCD')), pd.DataFrame({'x': [3, 4, 5, np.nan]}, index=list('ABCD')))])
def test_add_frames(self, first, second, expected):
    tm.assert_frame_equal(first + second, expected)
    tm.assert_frame_equal(second + first, expected)