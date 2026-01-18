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
def test_dt_is_year_start_end():
    ser = pd.Series([datetime(year=2023, month=12, day=31, hour=3), datetime(year=2023, month=1, day=1, hour=3), datetime(year=2023, month=3, day=31, hour=3), None], dtype=ArrowDtype(pa.timestamp('us')))
    result = ser.dt.is_year_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)
    result = ser.dt.is_year_end
    expected = pd.Series([True, False, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)