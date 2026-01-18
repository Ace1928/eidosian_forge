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
@pytest.mark.parametrize('date_type', [32, 64])
def test_dt_to_pydatetime_date_error(date_type):
    ser = pd.Series([date(2022, 12, 31)], dtype=ArrowDtype(getattr(pa, f'date{date_type}')()))
    msg = 'The behavior of ArrowTemporalProperties.to_pydatetime is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pytest.raises(ValueError, match='to_pydatetime cannot be called with'):
            ser.dt.to_pydatetime()