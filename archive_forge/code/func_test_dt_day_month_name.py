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
@pytest.mark.parametrize('method, exp', [['day_name', 'Sunday'], ['month_name', 'January']])
def test_dt_day_month_name(method, exp, request):
    _require_timezone_database(request)
    ser = pd.Series([datetime(2023, 1, 1), None], dtype=ArrowDtype(pa.timestamp('ms')))
    result = getattr(ser.dt, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)