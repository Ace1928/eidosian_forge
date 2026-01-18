from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_format_native_types(self, unit, dtype, dta_dti):
    dta, dti = dta_dti
    res = dta._format_native_types()
    exp = dti._data._format_native_types()
    tm.assert_numpy_array_equal(res, exp)