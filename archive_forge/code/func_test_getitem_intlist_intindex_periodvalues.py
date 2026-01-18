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
def test_getitem_intlist_intindex_periodvalues(self):
    ser = Series(period_range('2000-01-01', periods=10, freq='D'))
    result = ser[[2, 4]]
    exp = Series([pd.Period('2000-01-03', freq='D'), pd.Period('2000-01-05', freq='D')], index=[2, 4], dtype='Period[D]')
    tm.assert_series_equal(result, exp)
    assert result.dtype == 'Period[D]'