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
@pytest.mark.parametrize('method', ['ceil', 'floor', 'round'])
def test_dt_roundlike_tz_options_not_supported(method):
    ser = pd.Series([datetime(year=2023, month=1, day=2, hour=3), None], dtype=ArrowDtype(pa.timestamp('ns')))
    with pytest.raises(NotImplementedError, match='ambiguous is not supported.'):
        getattr(ser.dt, method)('1h', ambiguous='NaT')
    with pytest.raises(NotImplementedError, match='nonexistent is not supported.'):
        getattr(ser.dt, method)('1h', nonexistent='NaT')