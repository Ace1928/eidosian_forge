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
def test_dt_isocalendar():
    ser = pd.Series([datetime(year=2023, month=1, day=2, hour=3), None], dtype=ArrowDtype(pa.timestamp('ns')))
    result = ser.dt.isocalendar()
    expected = pd.DataFrame([[2023, 1, 1], [0, 0, 0]], columns=['year', 'week', 'day'], dtype='int64[pyarrow]')
    tm.assert_frame_equal(result, expected)