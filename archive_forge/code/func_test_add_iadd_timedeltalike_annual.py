import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_add_iadd_timedeltalike_annual(self):
    rng = period_range('2014', '2024', freq='Y')
    result = rng + pd.offsets.YearEnd(5)
    expected = period_range('2019', '2029', freq='Y')
    tm.assert_index_equal(result, expected)
    rng += pd.offsets.YearEnd(5)
    tm.assert_index_equal(rng, expected)