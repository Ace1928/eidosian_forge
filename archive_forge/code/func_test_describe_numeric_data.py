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
@pytest.mark.parametrize('pa_type', tm.ALL_INT_PYARROW_DTYPES + tm.FLOAT_PYARROW_DTYPES)
def test_describe_numeric_data(pa_type):
    data = pd.Series([1, 2, 3], dtype=ArrowDtype(pa_type))
    result = data.describe()
    expected = pd.Series([3, 2, 1, 1, 1.5, 2.0, 2.5, 3], dtype=ArrowDtype(pa.float64()), index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    tm.assert_series_equal(result, expected)