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
@pytest.mark.parametrize('scalar', ['foo', Timestamp('2021-01-01'), Timedelta(days=4), 9, 9.5, 2000, False, None])
def test_compare_invalid_scalar(self, box_with_array, scalar):
    pi = period_range('2000', periods=4)
    parr = tm.box_expected(pi, box_with_array)
    assert_invalid_comparison(parr, scalar, box_with_array)