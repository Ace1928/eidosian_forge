from datetime import timedelta
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(reason="Chained inequality raises when trying to define 'selector'")
def test_logical_range_select(self, datetime_series):
    selector = -0.5 <= datetime_series <= 0.5
    expected = (datetime_series >= -0.5) & (datetime_series <= 0.5)
    tm.assert_series_equal(selector, expected)