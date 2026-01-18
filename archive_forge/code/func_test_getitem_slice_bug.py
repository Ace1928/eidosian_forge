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
def test_getitem_slice_bug(self):
    ser = Series(range(10), index=list(range(10)))
    result = ser[-12:]
    tm.assert_series_equal(result, ser)
    result = ser[-7:]
    tm.assert_series_equal(result, ser[3:])
    result = ser[:-12]
    tm.assert_series_equal(result, ser[:0])