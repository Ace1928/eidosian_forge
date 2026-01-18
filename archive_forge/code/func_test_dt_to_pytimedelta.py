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
def test_dt_to_pytimedelta():
    data = [timedelta(1, 2, 3), timedelta(1, 2, 4)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.duration('ns')))
    result = ser.dt.to_pytimedelta()
    expected = np.array(data, dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    assert all((type(res) is timedelta for res in result))
    expected = ser.astype('timedelta64[ns]').dt.to_pytimedelta()
    tm.assert_numpy_array_equal(result, expected)