from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
@pytest.mark.parametrize('unit', ['us', 'ns'])
def test_dt_time_preserve_unit(unit):
    ser = pd.Series([datetime(year=2023, month=1, day=2, hour=3), None], dtype=ArrowDtype(pa.timestamp(unit)))
    assert ser.dt.unit == unit
    result = ser.dt.time
    expected = pd.Series(ArrowExtensionArray(pa.array([time(3, 0), None], type=pa.time64(unit))))
    tm.assert_series_equal(result, expected)