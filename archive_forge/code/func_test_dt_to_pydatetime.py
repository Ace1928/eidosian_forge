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
def test_dt_to_pydatetime():
    data = [datetime(2022, 1, 1), datetime(2023, 1, 1)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.timestamp('ns')))
    msg = 'The behavior of ArrowTemporalProperties.to_pydatetime is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser.dt.to_pydatetime()
    expected = np.array(data, dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    assert all((type(res) is datetime for res in result))
    msg = 'The behavior of DatetimeProperties.to_pydatetime is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = ser.astype('datetime64[ns]').dt.to_pydatetime()
    tm.assert_numpy_array_equal(result, expected)