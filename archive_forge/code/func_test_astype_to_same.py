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
def test_astype_to_same(self):
    arr = DatetimeArray._from_sequence(['2000'], dtype=DatetimeTZDtype(tz='US/Central'))
    result = arr.astype(DatetimeTZDtype(tz='US/Central'), copy=False)
    assert result is arr