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
def test_comparators(self, comparison_op):
    index = date_range('2020-01-01', periods=10)
    element = index[len(index) // 2]
    element = Timestamp(element).to_datetime64()
    arr = np.array(index)
    arr_result = comparison_op(arr, element)
    index_result = comparison_op(index, element)
    assert isinstance(index_result, np.ndarray)
    tm.assert_numpy_array_equal(arr_result, index_result)