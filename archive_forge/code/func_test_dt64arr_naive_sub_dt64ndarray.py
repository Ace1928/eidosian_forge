from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_dt64arr_naive_sub_dt64ndarray(self, box_with_array):
    dti = date_range('2016-01-01', periods=3, tz=None)
    dt64vals = dti.values
    dtarr = tm.box_expected(dti, box_with_array)
    expected = dtarr - dtarr
    result = dtarr - dt64vals
    tm.assert_equal(result, expected)
    result = dt64vals - dtarr
    tm.assert_equal(result, expected)