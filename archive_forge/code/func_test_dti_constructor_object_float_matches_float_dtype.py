from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_dti_constructor_object_float_matches_float_dtype(self):
    arr = np.array([0, np.nan], dtype=np.float64)
    arr2 = arr.astype(object)
    dti1 = DatetimeIndex(arr, tz='CET')
    dti2 = DatetimeIndex(arr2, tz='CET')
    tm.assert_index_equal(dti1, dti2)