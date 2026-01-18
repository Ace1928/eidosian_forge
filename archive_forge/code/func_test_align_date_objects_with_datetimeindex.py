from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_align_date_objects_with_datetimeindex(self):
    rng = date_range('1/1/2000', periods=20)
    ts = Series(np.random.default_rng(2).standard_normal(20), index=rng)
    ts_slice = ts[5:]
    ts2 = ts_slice.copy()
    ts2.index = [x.date() for x in ts2.index]
    result = ts + ts2
    result2 = ts2 + ts
    expected = ts + ts[5:]
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)