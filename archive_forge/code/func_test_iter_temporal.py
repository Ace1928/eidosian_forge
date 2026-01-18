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
@pytest.mark.parametrize('pa_type', tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES)
def test_iter_temporal(pa_type):
    arr = ArrowExtensionArray(pa.array([1, None], type=pa_type))
    result = list(arr)
    if pa.types.is_duration(pa_type):
        expected = [pd.Timedelta(1, unit=pa_type.unit).as_unit(pa_type.unit), pd.NA]
        assert isinstance(result[0], pd.Timedelta)
    else:
        expected = [pd.Timestamp(1, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit), pd.NA]
        assert isinstance(result[0], pd.Timestamp)
    assert result[0].unit == expected[0].unit
    assert result == expected