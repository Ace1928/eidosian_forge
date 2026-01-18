from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('freq_depr', ['2ye-mar', '2ys', '2qe', '2qs-feb', '2bqs', '2sms', '2bms', '2cbme', '2me', '2w'])
def test_date_range_lowercase_frequency_deprecated(self, freq_depr):
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version, please use '{freq_depr.upper()[1:]}' instead."
    expected = pd.date_range('1/1/2000', periods=4, freq=freq_depr.upper())
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = pd.date_range('1/1/2000', periods=4, freq=freq_depr)
    tm.assert_index_equal(result, expected)