from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_boolean_different_order(self, string_series):
    ordered = string_series.sort_values()
    sel = string_series[ordered > 0]
    exp = string_series[string_series > 0]
    tm.assert_series_equal(sel, exp)