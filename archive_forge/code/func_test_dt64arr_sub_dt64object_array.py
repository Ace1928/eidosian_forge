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
def test_dt64arr_sub_dt64object_array(self, box_with_array, tz_naive_fixture):
    dti = date_range('2016-01-01', periods=3, tz=tz_naive_fixture)
    expected = dti - dti
    obj = tm.box_expected(dti, box_with_array)
    expected = tm.box_expected(expected, box_with_array).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        result = obj - obj.astype(object)
    tm.assert_equal(result, expected)