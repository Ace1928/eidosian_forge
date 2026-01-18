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
def test_dt64arr_aware_sub_dt64ndarray_raises(self, tz_aware_fixture, box_with_array):
    tz = tz_aware_fixture
    dti = date_range('2016-01-01', periods=3, tz=tz)
    dt64vals = dti.values
    dtarr = tm.box_expected(dti, box_with_array)
    msg = 'Cannot subtract tz-naive and tz-aware datetime'
    with pytest.raises(TypeError, match=msg):
        dtarr - dt64vals
    with pytest.raises(TypeError, match=msg):
        dt64vals - dtarr