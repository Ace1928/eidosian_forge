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
def test_dt_normalize():
    ser = pd.Series([datetime(year=2023, month=3, day=30), datetime(year=2023, month=4, day=1, hour=3), datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59), None], dtype=ArrowDtype(pa.timestamp('us')))
    result = ser.dt.normalize()
    expected = pd.Series([datetime(year=2023, month=3, day=30), datetime(year=2023, month=4, day=1), datetime(year=2023, month=2, day=3), None], dtype=ArrowDtype(pa.timestamp('us')))
    tm.assert_series_equal(result, expected)