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
def test_dt_components():
    ser = pd.Series([pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4), None], dtype=ArrowDtype(pa.duration('ns')))
    result = ser.dt.components
    expected = pd.DataFrame([[1, 0, 0, 2, 0, 3, 4], [None, None, None, None, None, None, None]], columns=['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds'], dtype='int32[pyarrow]')
    tm.assert_frame_equal(result, expected)