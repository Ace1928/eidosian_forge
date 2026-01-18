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
@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
def test_dt64arr_addsub_time_objects_raises(self, box_with_array, tz_naive_fixture):
    tz = tz_naive_fixture
    obj1 = date_range('2012-01-01', periods=3, tz=tz)
    obj2 = [time(i, i, i) for i in range(3)]
    obj1 = tm.box_expected(obj1, box_with_array)
    obj2 = tm.box_expected(obj2, box_with_array)
    msg = '|'.join(['unsupported operand', 'cannot subtract DatetimeArray from ndarray'])
    assert_invalid_addsub_type(obj1, obj2, msg=msg)